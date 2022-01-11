from __future__ import annotations
from typing import cast, Tuple, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.logger import getArkoudaLogger
#import numpy as np 
from arkouda.dtypes import npstr as akstr
from arkouda.dtypes import int64 as akint
from arkouda.dtypes import NUMBER_FORMAT_STRINGS, resolve_scalar_dtype, \
     translate_np_dtype
import json

__all__ = ["Graph",
           "rmat_gen","graph_file_read",
           "graph_bfs"]



class Graph:
    """
    This is a double index graph data structure based graph representation. The graph data resides on the
    arkouda server. The user should not call this class directly;
    rather its instances are created by other arkouda functions.

    Attributes
    ----------
    n_vertices : int
        The starting indices for each string
    n_edges : int
        The starting indices for each string
    directed : int
        The graph is directed (True) or undirected (False)
    weighted : int
        The graph is weighted (True) or not
    name : string
        The graph name in Chapel
    logger : ArkoudaLogger
        Used for all logging operations
        
    Notes
    -----
    """

    def __init__(self, *args) -> None:
        """
        Initializes the Graph instance by setting all instance
        attributes, some of which are derived from the array parameters.
        
        Parameters
        ----------
        n_vertices  : must provide args[0]
        n_edges     : must provide args[1]
        directed    : must provide args[2]
        weighted    : must provide args[3]
        name        : must provide args[4]
        
            
        Returns
        -------
        None
        
        Raises
        ------
        RuntimeError
            Raised if there's an error converting a Numpy array or standard
            Python array to either the offset_attrib or bytes_attrib   
        ValueError
            Raised if there's an error in generating instance attributes 
            from either the offset_attrib or bytes_attrib parameter 
        """
        try:
            if len(args) < 5: 
                raise ValueError
            self.n_vertices=cast(int,args[0])
            self.n_edges=cast(int,args[1])
            self.directed=cast(int,args[2])
            self.weighted=cast(int,args[3])
            self.name=cast(str,args[4])
        except Exception as e:
            raise RuntimeError(e)
        
        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __size__(self) -> int:
        return self.n_vertices

@typechecked
def graph_file_read(Ne:int, Nv:int,Ncol:int,directed:int, filename: str)  -> Graph:
        """
        This function is used for creating a graph from a file.
        The file should like this
          1   5
          13  9
          4   8
          7   6
        This file means the edges are <1,5>,<13,9>,<4,8>,<7,6>. If additional column is added, it is the weight
        of each edge.
        Ne : the total number of edges of the graph
        Nv : the total number of vertices of the graph
        Ncol: how many column of the file. Ncol=2 means just edges (so no weight and weighted=0) 
              and Ncol=3 means there is weight for each edge (so weighted=1). 
        directed: 0 means undirected graph and 1 means directed graph
        Returns
        -------
        Graph
            The Graph class to represent the data

        See Also
        --------

        Notes
        -----
        
        Raises
        ------  
        RuntimeError
        """
        cmd = "segmentedGraphFile"
        RCMFlag=0
        DegreeSortFlag=0
        args="{} {} {} {} {} {} {}".format(Ne, Nv, Ncol,directed, filename,RCMFlag,DegreeSortFlag);
        repMsg = generic_msg(cmd=cmd,args=args)
        if (int(Ncol) >2) :
             weighted=1
        else:
             weighted=0

        return Graph(*(cast(str,repMsg).split('+')))


@typechecked
def rmat_gen (lgNv:int, Ne_per_v:int, p:float, directed: int,weighted:int) ->Graph:
        """
        This function is for creating a graph using rmat graph generator
        Returns
        -------
        Graph
            The Graph class to represent the data

        See Also
        --------

        Notes
        -----
        
        Raises
        ------  
        RuntimeError
        """
        cmd = "segmentedRMAT"
        RCMFlag=1
        args= "{} {} {} {} {} {}".format(lgNv, Ne_per_v, p, directed, weighted,RCMFlag)
        msg = "segmentedRMAT {} {} {} {} {}".format(lgNv, Ne_per_v, p, directed, weighted)
        repMsg = generic_msg(cmd=cmd,args=args)

        return Graph(*(cast(str,repMsg).split('+')))

@typechecked
def graph_bfs (graph: Graph, root: int ) -> pdarray:
        """
        This function is generating the breadth-first search vertices sequences in given graph
        starting from the given root vertex
        Returns
        -------
        pdarray
            The bfs vertices results

        See Also
        --------

        Notes
        -----
        
        Raises
        ------  
        RuntimeError
        """
        cmd="segmentedGraphBFS"
        DefaultRatio=0.9
        RCMFlag=1
        args = "{} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.name,
                 root,DefaultRatio)
        repMsg = generic_msg(cmd=cmd,args=args)
        return create_pdarray(repMsg)

