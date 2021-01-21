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

__all__ = ['Graph']

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
        return self.adjacency.size


    def __str__(self) -> str:
        return "vertex id={},#adjacency={},weight={}".format(self.vertex_id,\
                self.size,self.weight)

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

    def __size__(self) -> int:
        return self.adjacency.size/2


    def __str__(self) -> str:
        return "vertex pair={},#adjacency={},weight={}".format(self.vertex_pair,\
                self.size,self.weight)

    def __repr__(self) -> str:
        return "{}".format(self.__str__())


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
    directed : bool
        The raw bytes of all strings, joined by nulls
    src : pdarray
        The source of every edge in the graph
    dst : pdarray
        The destination of every vertex in the graph
    v_weight : pdarray
        The weitht of every vertex in the graph
    e_weight : pdarray
        The weitht of every edge in the graph
    neighbour : pdarray
        The current vertix id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
        if v=n_vertices-1, then v's neighbours are from dst[neighbour[v+1]] to dst[n_edges-1]
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
        neighbour   : optional if no src and dst
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
            if len(args) > 7:
                if isinstance(args[7],pdarray):
                     self.e_weight=args[7]
                else:
                    try:
                        self.e_weight = create_pdarray(args[7])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 6:
                if isinstance(args[6],pdarray):
                     self.v_weight=args[6]
                else:
                    try:
                        self.v_weight = create_pdarray(args[6])
                    except Exception as e:
                        raise RuntimeError(e)
            if len(args) > 5:
                if isinstance(args[5],pdarray):
                     self.neighbour=args[5]
                else:
                    try:
                        self.neighbour = create_pdarray(args[5])
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
                     self.directed=cast(bool,args[2])
        except Exception as e:
            raise RuntimeError(e)
        self.dtype = akint
        self.logger = getArkoudaLogger(name=__class__.__name__) # type: ignore

    def __iter__(self):
        raise NotImplementedError('Graph does not support iteration')

    def __size__(self) -> int:
        return self.n_vertices


    def add_vertice(self, x: Vertice) :
        print()

    def remove_vertice(self, x: int) :
        print()

    def neighbours(self, x: int)->pdarray :
        print()

    def adjacent(self, x: int, y:int )->pdarray :
        print()

    def get_vertice_value(self, x: int) -> Vertice:
        print()

    def set_vertice_value(self, x: int, v: Vertice) :
        print()

    def add_edge(self, x: int, y: int) :
        print()

    def remove_edge(self, x: int, y: int) :
        print()

