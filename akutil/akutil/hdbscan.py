import pandas as pd
import numpy as np
import akutil as aku
import arkouda as ak
import math
from collections import defaultdict
from tqdm.auto import tqdm

__all__ = [
        "Level",
        "Clusterer",
        "rescale_weights",
    ]

class Clusterer:
    """ An implementation of the HDBSCAN hierarchical clustering algorithm
    specific to `arkouda` arrays and for use on a large distributed memory
    architecture.

    level_data is a list of Levels that indicates which nodes are in the
    same connected component at that level (i.e., distance).

    level_data should be sorted by delta, from smallest to largest.


    See also:
        https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

    References:
        Leland McInnes, John Healy, and Steve Astels
        Journal of Open Source Software
        _hdbscan: Hierarchical density based clustering_
        Volume 2, Number 11, March 2017
        DOI: 10.21105/joss.00205
    """

    def __init__(self, level_data):
        self.level_data = level_data

    def deselect_children(self, node=None):
        if not node:
            p = self.selection_data.size
        else:
            p = node
        children = self.selection_data['index'][self.selection_data['parent']==p]
        for child in children:
            if self.selection_data['selected'][child]:
                self.selection_data['selected'][child] = False
            else:
                self.deselect_children(node=child)

    # Maybe pass a vector along that says which nodes we deselect. Then we can
    # update those values when the process returns.

    def select_clusters(self):
        print("Computing Selection and Stability.")
        # Perhaps keep track of a "final clusters" array, that we update as we
        # work through this function.
        self.selection_data['selected'] = ak.ones(self.selection_data.size, dtype=ak.bool)
        byparent = ak.GroupBy(self.selection_data['parent'])
        uk = byparent.unique_keys
        for p in tqdm(uk[1:]):
            children = self.selection_data['index'][self.selection_data['parent'] == p]
            c_stab = (self.selection_data['stability'][children]).sum()
            p_stab = self.selection_data['stability'][p]
            if c_stab >= p_stab:
                self.selection_data['stability'][p] = c_stab
                self.selection_data['selected'][p] = False
            else:
                self.deselect_children(node=p)
        print("Selection and Stability computation is complete!")

    def extract_clusters(self):
        # List all the time window keys
        deltas = list(self.cluster_data.keys())
        # Reverse them so we can start with the last clustering data
        deltas.reverse()
        # Ignore delta 0 as it is an artifact of the clustering that isn't used
        deltas = deltas[:-1]

        print("Extracting clusters from each time delta: ".format(deltas))

        # This is list of cluster labels which we will update at each time delta
        # where a value of 0 should indicate an unclustered node
        final_cluster_info = ak.zeros_like(self.cluster_data[deltas[0]]['index'])
        # A list of cluster labels that are selected
        selected_clusters = self.selection_data['index'][self.selection_data['selected']]
        selected_clusters = selected_clusters[selected_clusters > 0]

        for delta in tqdm(deltas):
            cluster = self.cluster_data[delta]['labels']
            cluster_positive = ak.where(cluster < 0, -cluster, 0)

            # The cluster labels found in this delta
            labels_this_delta = cluster_positive[cluster_positive > 0]

            # A boolean array to indicate which "selected" clusters are labels this delta
            m = ak.in1d(selected_clusters, labels_this_delta)

            # A list of clusters selected for this delta
            extract_this_delta = selected_clusters[m]

            # A boolean array indicating which nodes are in clusters that are extracted this delta
            m2 = ak.in1d(cluster_positive, extract_this_delta)

            # Indicate the clusters for all the nodes in clusters that we extracted this delta
            final_cluster_info[m2] = cluster_positive[m2]
            v,c = ak.value_counts(cluster_positive[m2])
            selected_clusters = selected_clusters[(~m)]

        self.extracted_clusters = final_cluster_info

        if selected_clusters.size > 0:
            print("Failed. {} of the selected clusters remain.".format(selected_clusters.size))
            print("Failing cluster labels: {}".format(selected_clusters))

            # We can refer to this list here:
            self.unextracted = selected_clusters
        else:
            print("Extraction completed succesfully.")


    def cluster(self, min_cluster_size=5):
        cluster_data = {}
        last_level_delta = self.level_data[0].delta

        # Initial setup; all levels are the same size
        num_nodes = self.level_data[0].size

        # This dataframe holds extraction data
        selection_data = aku.DataFrame({
                'stability': ak.zeros(1, dtype=ak.float64),
                'parent': ak.zeros(1, dtype=ak.int64),
            })

        # Create an initial cluster dataframe
        labels = ak.arange(num_nodes)
        sizes = ak.ones(num_nodes, dtype=ak.int64)
        stability = ak.zeros(num_nodes, dtype=ak.float64)
        selected = ak.zeros(num_nodes, dtype=ak.bool)

        df = aku.DataFrame({
            'cc':self.level_data[0].cc,
            'labels':labels,
            'sizes':sizes,
            'stability':stability,
        })
        # The result should have all the same keys as the deltas
        cluster_data[self.level_data[0].delta] = df

        # We don't start with the level 0, it gets passed through as is.
        for level in tqdm(self.level_data[1:]):
            bylevel = ak.GroupBy(level.cc)
            perm = bylevel.permutation
            # Save for later analysis
            old_labels = labels[:]
            # Count number of nodes in each group
            _,c = bylevel.count()
            # Find largest (negative) label value each group
            _, max_group_labels = bylevel.aggregate(labels, 'min')
            # Find maximum of existing cluster sizes from last iteration.
            _, max_group_size = bylevel.aggregate(sizes, 'max')
            # Find the maximum stability in each group
            _, max_group_stability = bylevel.aggregate(stability, 'max')
            # Find the number of sub-clusters in each group for purposes of creating new cluster labels
            clusters_and_zeros = ak.where(labels < 0, labels, 0)
            _, num_unique_labels = bylevel.aggregate(clusters_and_zeros, 'nunique')
            _, min_group_label = bylevel.aggregate(labels, 'max')
            num_sub_clusters = num_unique_labels - ak.where(min_group_label >= 0, 1, 0)

            # Update sizes
            count_bc = bylevel.broadcast(c)
            sizes = ak.zeros(num_nodes, dtype=ak.int64)
            sizes[perm] = count_bc

            # Update labels to max (negative) in group
            labels_bc = bylevel.broadcast(max_group_labels)
            labels = ak.zeros(num_nodes, dtype=ak.int64)
            labels[perm] = labels_bc

            # Update stability
            stability_bc = bylevel.broadcast(max_group_stability)
            stability = ak.zeros(num_nodes, dtype=ak.float64)
            stability[perm] = stability_bc

            # Create and update labels as needed, baseline size is 1
            # Only need to test if there are at least two cluster labels in a group.
            new_clusters_join = (num_sub_clusters > 1)
            new_clusters_form = ((c >= min_cluster_size) & (max_group_labels >= 0))
            condition = (new_clusters_join | new_clusters_form)
            num_new_labels = int(condition.sum())

            new_labels_positioned = ak.zeros(c.size, dtype=np.int64)
            if num_new_labels > 0:
                # Set up selection_data 
                mn = abs(int(labels.min()))
                new_label_values = ak.arange(mn+1, mn+num_new_labels+1, 1) * (-1)
                new_labels_positioned = ak.zeros(c.size, dtype=np.int64)
                new_labels_positioned[condition] = new_label_values

                # Update selection_data
                update_df = aku.DataFrame({
                    'parent': ak.zeros(num_new_labels, dtype=ak.int64),
                    'stability': ak.zeros(num_new_labels, dtype=ak.float64),
                })
                selection_data.append(update_df)

                # Update the labels
                labels_bc = bylevel.broadcast(new_labels_positioned)
                new_labels = ak.zeros(num_nodes, dtype=ak.int64)
                new_labels[perm] = labels_bc
                tmp = ak.where(new_labels < 0, new_labels, labels)
                labels = tmp

                # When clusters become absorbed into new clusters, add their parent labels and update stability
                mask = ((labels < 0) & (old_labels < 0) & (labels < old_labels))
                if mask.sum() > 0:
                    t1 = old_labels[mask]
                    t2 = labels[mask]
                    t3 = stability[mask]
                    bychangedlabels = ak.GroupBy([t1, t2])
                    [old,new] = bychangedlabels.unique_keys
                    # I don't remember the purpose of this line, but it's never used.
                    #stabby = t3[aku.invert_permutation(bychangedlabels.permutation)][bychangedlabels.segments]
                    selection_data['parent'][-1 * old] = -1 * new

            # Set new cluster stability to 0
            new_label_bc = bylevel.broadcast(new_labels_positioned)
            tmp = ak.zeros(labels.size, dtype=np.int64)
            tmp[perm] = new_label_bc
            stability[tmp < 0] = 0

            # Update stability
            added_stability = sizes / (level.delta - last_level_delta)
            last_level_delta = level.delta
            tmp = ak.where(sizes >= min_cluster_size, stability + added_stability, stability)
            stability = tmp

            # Save this information after processing
            df = aku.DataFrame({
                'cc':level.cc,
                'labels':labels,
                'sizes':sizes,
                'stability':stability,
            })
            cluster_data[level.delta] = df

            # Update cluster selection information
            bylabel = ak.GroupBy(labels)
            keys = labels[bylabel.permutation][bylabel.segments]
            stab = stability[bylabel.permutation][bylabel.segments]
            indx = (keys[keys < 0])*(-1)
            vals = stab[keys < 0]
            selection_data['stability'][indx] = vals

        # Set up data for next steps
        self.cluster_data = cluster_data
        self.selection_data = selection_data

        # Select and extract
        self.select_clusters()
        self.extract_clusters()

        print("Clustering is complete!")

        return self.extracted_clusters


class Level:
    """ A simple container for level data and delta value."""
    def __init__(self, cc, delta):
        self.cc = cc
        self.delta = delta
        self.size = cc.size

    def __repr__(self):
        return "Level(<CC>, delta: " + str(self.delta) + ")"

def rescale_weights(float_arr, nsegments, method='intstep'):
    """ Rescale a list of waits into nsegments of discrete values in
    preparation for running through the HDBSCAN clustering algorithm.

    """

    nvalues = len(float_arr)
    mx = float_arr.max()
    mn = float_arr.min()
    if method == 'intstep':
        scale = np.linspace(mn, mx, nsegments, endpoint=False)
        res_arr = np.zeros(nvalues)
        for i in range(nsegments):
            res_arr[float_arr >= scale[i]] = i + 1
        return res_arr
    elif method == 'equal':
        p = np.argsort(float_arr)
        size = nvalues // nsegments
        res_arr = np.zeros(nvalues)
        for i in range(nsegments):
            res_arr[i*size:] = i
        return res_arr[np.argsort(p)]
    elif method == 'log':
        p = np.argsort(float_arr)
        wts = np.floor(sorted(10- np.logspace(0,1,nvalues)))
        return wts[np.argsort(p)]
