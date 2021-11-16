from __future__ import annotations
from typing import cast, Tuple, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.logger import getArkoudaLogger
import numpy as np 
from arkouda.dtypes import npstr as akstr
from arkouda.dtypes import int64 as akint
from arkouda.dtypes import NUMBER_FORMAT_STRINGS, resolve_scalar_dtype, \
     translate_np_dtype
import json

__all__ = ["GraphD","GraphDW","GraphUD","GraphUDW",
           "rmat_gen","graph_file_read", "stream_file_read",
           "graph_bfs",
           "graph_bc",
           "graph_triangle",
           "stream_tri_cnt","streamPL_tri_cnt",
           "KTruss" ]




class GraphD:
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
    src : pdarray
        The source of every edge in the graph
    dst : pdarray
        The destination of every vertex in the graph
    start_i : pdarray
        The starting index of all the vertices in src 
    neighbour : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
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
        src,dst     : optional if no weighted  args[4] args[5]
        start_i, neighbour   : optional if no src and dst args[6] args[7]
        v_weight    : optional if no neighbour  args[8] 
        e_weight    : optional if no v_weight   args[9]
        
            
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
            if len(args) < 4: 
                raise ValueError
            self.n_vertices=cast(int,args[0])
            self.n_edges=cast(int,args[1])
            self.directed=cast(int,args[2])
            self.weighted=cast(int,args[3])
            if len(args) == 7:
                raise ValueError
            if len(args) > 7:
                if (isinstance(args[7],pdarray) and isinstance(args[6],pdarray)) :
                     self.start_i=args[6]
                     self.neighbour=args[7]
                else:
                    try:
                        self.start_i = create_pdarray(args[6])
                        self.neighbour = create_pdarray(args[7])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) == 5:
                raise ValueError
            if len(args) > 5:
                if (isinstance(args[5],pdarray) and isinstance(args[4],pdarray)) :
                     self.src=args[4]
                     self.dst=args[5]
                else:
                    try:
                        self.src = create_pdarray(args[4])
                        self.dst = create_pdarray(args[5])
                    except Exception as e:
                        raise RuntimeError(e)
        except Exception as e:
            raise RuntimeError(e)
        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __size__(self) -> int:
        return self.n_vertices


class GraphDW(GraphD):
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
    src : pdarray
        The source of every edge in the graph
    dst : pdarray
        The destination of every vertex in the graph
    start_i : pdarray
        The starting index of all the vertices in src 
    neighbour : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
    v_weight : pdarray
        The weitht of every vertex in the graph
    e_weight : pdarray
        The weitht of every edge in the graph
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
        src,dst     : optional if no weighted  args[4] args[5]
        start_i, neighbour   : optional if no src and dst args[6] args[7]
        v_weight    : optional if no neighbour  args[8] 
        e_weight    : optional if no v_weight   args[9]
        
            
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
        super().__init__(*args)
        try:
            if len(args) > 9:
                if isinstance(args[9],pdarray):
                     self.e_weight=args[9]
                else:
                    try:
                        self.e_weight = create_pdarray(args[9])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 8:
                if isinstance(args[8],pdarray):
                     self.v_weight=args[8]
                else:
                    try:
                        self.v_weight = create_pdarray(args[8])
                    except Exception as e:
                        raise RuntimeError(e)
        except Exception as e:
            raise RuntimeError(e)



class GraphUD(GraphD):
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
    src : pdarray
        The source of every edge in the graph
    dst : pdarray
        The destination of every vertex in the graph
    start_i : pdarray
        The starting index of all the vertices in src 
    neighbour : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
    srcR : pdarray
        The source of every edge in the graph
    dstR : pdarray
        The destination of every vertex in the graph
    start_iR : pdarray
        The starting index of all the vertices in src 
    neighbourR : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
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
        src,dst     : optional if no weighted  args[4] args[5]
        start_i, neighbour   : optional if no src and dst args[6] args[7]
        srcR,dstR     : optional if no neighbour  args[8] args[9]
        start_iR, neighbourR   : optional if no dstR  args[10] args[11]
        v_weight    : optional if no neighbouirR args[12] 
        e_weight    : optional if no v_weight    args[13]

        Returns
        -------
        None
        
        Raises
        ------
        RuntimeError
        ValueError
        """
        super().__init__(*args)
        try:
            if len(args) > 11:
                if (isinstance(args[11],pdarray) and isinstance(args[10],pdarray)) :
                     self.start_iR=args[10]
                     self.neighbourR=args[11]
                else:
                    try:
                        self.start_iR = create_pdarray(args[10])
                        self.neighbourR = create_pdarray(args[11])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 9:
                if (isinstance(args[9],pdarray) and isinstance(args[8],pdarray)) :
                     self.srcR=args[8]
                     self.dstR=args[9]
                else:
                    try:
                        self.srcR = create_pdarray(args[8])
                        self.dstR = create_pdarray(args[9])
                    except Exception as e:
                        raise RuntimeError(e)
        except Exception as e:
            raise RuntimeError(e)


class GraphUDW(GraphUD):
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
    src : pdarray
        The source of every edge in the graph
    dst : pdarray
        The destination of every vertex in the graph
    start_i : pdarray
        The starting index of all the vertices in src 
    neighbour : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
    srcR : pdarray
        The source of every edge in the graph
    dstR : pdarray
        The destination of every vertex in the graph
    start_iR : pdarray
        The starting index of all the vertices in src 
    neighbourR : pdarray
        The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
        neighbour[v+1]-neighbour[v] if v<n_vertices-1 or 
        n_edges-neighbour[v] if v=n_vertices-1
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
        src,dst     : optional if no weighted  args[4] args[5]
        start_i, neighbour   : optional if no src and dst args[6] args[7]
        srcR,dstR     : optional if no neighbour  args[8] args[9]
        start_iR, neighbourR   : optional if no dstR  args[10] args[11]
        v_weight    : optional if no neighbouirR args[12] 
        e_weight    : optional if no v_weight    args[13]
        
            
        Returns
        -------
        None
        
        Raises
        ------
        RuntimeError
        ValueError
        """
        super().__init__(*args)
        try:
            if len(args) > 13:
                if isinstance(args[13],pdarray):
                     self.e_weight=args[13]
                else:
                    try:
                        self.e_weight = create_pdarray(args[13])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 12:
                if isinstance(args[12],pdarray):
                     self.v_weight=args[12]
                else:
                    try:
                        self.v_weight = create_pdarray(args[12])
                    except Exception as e:
                        raise RuntimeError(e)
        except Exception as e:
            raise RuntimeError(e)



@typechecked
def graph_file_read(Ne:int, Nv:int,Ncol:int,directed:int, filename: str)  -> Union[GraphD,GraphUD,GraphDW,GraphUDW]:
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

        if (directed!=0)  :
           if (weighted!=0) :
               return GraphDW(*(cast(str,repMsg).split('+')))
           else:
               return GraphD(*(cast(str,repMsg).split('+')))
        else:
           if (weighted!=0) :
               return GraphUDW(*(cast(str,repMsg).split('+')))
           else:
               return GraphUD(*(cast(str,repMsg).split('+')))


@typechecked
def rmat_gen (lgNv:int, Ne_per_v:int, p:float, directed: int,weighted:int) ->\
              Union[GraphD,GraphUD,GraphDW,GraphUDW]:
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
        if (directed!=0)  :
           if (weighted!=0) :
               return GraphDW(*(cast(str,repMsg).split('+')))
           else:
               return GraphD(*(cast(str,repMsg).split('+')))
        else:
           if (weighted!=0) :
               return GraphUDW(*(cast(str,repMsg).split('+')))
           else:
               return GraphUD(*(cast(str,repMsg).split('+')))

@typechecked
def graph_bfs (graph: Union[GraphD,GraphDW,GraphUD,GraphUDW], root: int ) -> pdarray:
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
        if (int(graph.directed)>0)  :
            if (int(graph.weighted)==0):
              args = "{} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 root,DefaultRatio)
            else:
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.v_weight.name,graph.e_weight.name,\
                 root,DefaultRatio)
        else:
            if (int(graph.weighted)==0):
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.srcR.name,graph.dstR.name,\
                 graph.start_iR.name,graph.neighbourR.name,\
                 root,DefaultRatio)
            else:
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.srcR.name,graph.dstR.name,\
                 graph.start_iR.name,graph.neighbourR.name,\
                 graph.v_weight.name,graph.e_weight.name,\
                 root,DefaultRatio)

        repMsg = generic_msg(cmd=cmd,args=args)
        return create_pdarray(repMsg)

@typechecked
def graph_bfs (graph: Union[GraphD,GraphDW,GraphUD,GraphUDW], root: int ) -> pdarray:
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
        DefaultRatio=-.60
        RCMFlag=0
        if (int(graph.directed)>0)  :
            if (int(graph.weighted)==0):
              args = "{} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 root,DefaultRatio)
            else:
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.v_weight.name,graph.e_weight.name,\
                 root,DefaultRatio)
        else:
            if (int(graph.weighted)==0):
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.srcR.name,graph.dstR.name,\
                 graph.start_iR.name,graph.neighbourR.name,\
                 root,DefaultRatio)
            else:
              args = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                 RCMFlag,\
                 graph.n_vertices,graph.n_edges,\
                 graph.directed,graph.weighted,\
                 graph.src.name,graph.dst.name,\
                 graph.start_i.name,graph.neighbour.name,\
                 graph.srcR.name,graph.dstR.name,\
                 graph.start_iR.name,graph.neighbourR.name,\
                 graph.v_weight.name,graph.e_weight.name,\
                 root,DefaultRatio)

        repMsg = generic_msg(cmd=cmd,args=args)
        return create_pdarray(repMsg)

