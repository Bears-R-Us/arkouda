from __future__ import annotations
from typing import cast, Tuple, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
#, parse_single_value,_parse_single_int_array_value
from arkouda.logger import getArkoudaLogger
import numpy as np # type: ignore
from arkouda.dtypes import str as akstr
from arkouda.dtypes import int64 as akint
from arkouda.dtypes import NUMBER_FORMAT_STRINGS, resolve_scalar_dtype, \
     translate_np_dtype
import json

__all__ = ['Graph','GraphD','GraphDW','GraphUD','GraphUDW']

class Vertex:
    """
    Represents a vertex of a graph

    Attributes
    ----------
    vertex_id : int
        The unique identification of the vertex in a graph
    weight : int
        The weitht information of the current vertex
    neighbors  : pdarray
        all the vertices connected to the current vertex. For directed graph, out edge vertices are given.
    logger : ArkoudaLogger
        Used for all logging operations

    Notes
    -----
    Vertex is composed of one pdarray: the ID value array which 
    contains the all the ids of the adjacency vertices.
    """
    # based on args 
    def __init__(self, *args) -> None: 
        
        try:
            self.vertex_id=args[0]
            if len(args) > 2: 
                if isinstance(args[2],pdarray):
                     self.neighbours=args[2]
                else:
                    try:
                        self.neighbours = create_pdarray(args[2])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 1: 
                self.weight=args[1]
        except Exception as e:
            raise RuntimeError(e)

        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __size__(self) -> int:
        return self.neighbours.size


    def __str__(self) -> str:
        return "vertex id={},weight={},#neighbours={}".format(self.vertex_id,\
                self.weight, self.neighbours.size)

    def __repr__(self) -> str:
        return "{}".format(self.__str__())


class Edge:
    """
    Represents an Edge of a graph

    Attributes
    ----------
    vertex_pair : tuple
        The unique identification of the edge in a graph
    weight : int
        The weitht information of the current edge
    adjacency  : pdarray
        all the vertices connected to the current vertex. For directed graph, out edge vertices are given.
    logger : ArkoudaLogger
        Used for all logging operations

    Notes
    -----
    Vertex is composed of one pdarray: the ID value array which 
    contains the all the ids of the adjacency vertices.
    """
    # based on args 
    def __init__(self, *args) -> None:
        try:
            self.vertex_pair=args[0]
            if len(args) > 2:
                if isinstance(args[2],pdarray):
                     self.adjacency=args[2]
                else:
                    try:
                        self.adjacency = create_pdarray(args[2])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 1:
                self.weight=args[1]
        except Exception as e:
            raise RuntimeError(e)
        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __str__(self) -> str:
        return "vertex pair={},weight={},#adjacency={}".format(self.vertex_pair,\
                self.weight,self.adjacency.size)

    def __repr__(self) -> str:
        return "{}".format(self.__str__())


class GraphD:
    """
    This is an array based graph representation. The graph data resides on the
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

    '''
    def add_vertice(self, x: Vertice)->None :
        print()

    def remove_vertice(self, x: int) ->None:
        print()

    def neighbours(self, x: int)->pdarray :
        return self.neighbour[i]

    def adjacent(self, x: int, y:int )->pdarray :
        neighbours(self,x)
        neighbours(self,y)

    def get_vertice_value(self, x: int) -> Vertice:
        print()

    def set_vertice_value(self, x: int, v: Vertice) :
        print()

    def add_edge(self, x: int, y: int) :
        print()

    def remove_edge(self, x: int, y: int) :
        print()
    '''

class GraphDW(GraphD):
    """
    This is an array based graph representation. The graph data resides on the
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
    This is an array based graph representation. The graph data resides on the
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
    This is an array based graph representation. The graph data resides on the
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


class Graph:
    """
    This is an array based graph representation. The graph data resides on the
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
        n_vertices  : must provide
        n_edges     : must provide 
        directed    : optional  
        src,dst     : optional if no directed
        start_i, neighbour   : optional if no src and dst
        v_weight    : optional if no neighbour
        e_weight    : optional if no v_weight
        
            
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
            if len(args) < 2: 
                raise ValueError
            self.n_vertices=cast(int,args[0])
            self.n_edges=cast(int,args[1])
            if len(args) > 8:
                if isinstance(args[8],pdarray):
                     self.e_weight=args[8]
                else:
                    try:
                        self.e_weight = create_pdarray(args[8])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 7:
                if isinstance(args[7],pdarray):
                     self.v_weight=args[7]
                else:
                    try:
                        self.v_weight = create_pdarray(args[7])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) == 6:
                raise ValueError
            if len(args) > 6:
                if (isinstance(args[6],pdarray) and isinstance(args[5],pdarray)) :
                     self.start_i=args[5]
                     self.neighbour=args[6]
                else:
                    try:
                        self.start_i = create_pdarray(args[5])
                        self.neighbour = create_pdarray(args[6])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) == 4:
                raise ValueError
            if len(args) > 4:
                if (isinstance(args[4],pdarray) and isinstance(args[3],pdarray)) :
                     self.src=args[3]
                     self.dst=args[4]
                else:
                    try:
                        self.src = create_pdarray(args[3])
                        self.dst = create_pdarray(args[4])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 2:
                     self.directed=cast(int,args[2])
        except Exception as e:
            raise RuntimeError(e)
        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __size__(self) -> int:
        return self.n_vertices

    '''
    def add_vertice(self, x: Vertice)->None :
        print()

    def remove_vertice(self, x: int) ->None:
        print()

    def neighbours(self, x: int)->pdarray :
        return self.neighbour[i]

    def adjacent(self, x: int, y:int )->pdarray :
        neighbours(self,x)
        neighbours(self,y)

    def get_vertice_value(self, x: int) -> Vertice:
        print()

    def set_vertice_value(self, x: int, v: Vertice) :
        print()

    def add_edge(self, x: int, y: int) :
        print()

    def remove_edge(self, x: int, y: int) :
        print()
    '''



