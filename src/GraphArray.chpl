module GraphArray {

    use AryUtil;
    use MultiTypeSymEntry;
    use ServerConfig;
    use Reflection;
    use Logging;
    use ServerErrors;
    use NumPyDType;
    use Map;

    private config const logLevel = ServerConfig.logLevel;
    const graphLogger = new Logger(logLevel);

    // These are the component Key names stored in our components map
    enum Component {
        SRC,          // The source of every edge in the graph,array value
        SRC_R,        // Reverse of SRC
        DST,          // The destination of every vertex in the graph,array value
        DST_R,        // Reverse of DST
        START_IDX,    // The starting index of every vertex in src and dst
        START_IDX_R,  // Reverse of START_IDX
        NEIGHBOR,     // Numer of neighbors for a vertex  
        NEIGHBOR_R,   // 
        EDGE_WEIGHT,  // Edge weight
        VERTEX_WEIGHT // Vertex weight
    }

    /**
    * We use several arrays and intgers to represent a graph 
    * Instances are ephemeral, not stored in the symbol table. Instead, attributes
    * of this class refer to symbol table entries that persist. This class is a
    * convenience for bundling those persistent objects and defining graph-relevant
    * operations.
    * Now we  copy from SegSArray, we need change more in the future to fit a graph
    */
    class SegGraph {

        /* Map to hold various components of our Graph; use enum Component values as map keys */
        var components = new map(Component, shared GenSymEntry, parSafe=false);

        /* Total number of vertices */
        var n_vertices : int;

        /* Total number of edges */
        var n_edges : int;

        /* The graph is directed (True) or undirected (False)*/
        var directed : bool;

        /**
        * Init the basic graph object, we'll compose the pieces in
        * using the withCOMPONENT methods.
        */
        proc init(num_v:int, num_e:int, directed:bool) {
            this.n_vertices = num_v;
            this.n_edges = num_e;
            this.directed = directed;
        }

        proc isDirected():bool { return this.directed; }

        /* Use the withCOMPONENT methods to compose the graph object */
        proc withSRC(a:shared GenSymEntry):SegGraph { components.add(Component.SRC, a); return this; }
        proc withSRC_R(a:shared GenSymEntry):SegGraph { components.add(Component.SRC_R, a); return this; }
        
        proc withDST(a:shared GenSymEntry):SegGraph { components.add(Component.DST, a); return this; }
        proc withDST_R(a:shared GenSymEntry):SegGraph { components.add(Component.DST_R, a); return this; }
        
        proc withSTART_IDX(a:shared GenSymEntry):SegGraph { components.add(Component.START_IDX, a); return this; }
        proc withSTART_IDX_R(a:shared GenSymEntry):SegGraph { components.add(Component.START_IDX_R, a); return this; }

        proc withNEIGHBOR(a:shared GenSymEntry):SegGraph { components.add(Component.NEIGHBOR, a); return this; }
        proc withNEIGHBOR_R(a:GenSymEntry):SegGraph { components.add(Component.NEIGHBOR_R, a); return this; }

        proc withEDGE_WEIGHT(a:shared GenSymEntry):SegGraph { components.add(Component.EDGE_WEIGHT, a); return this; }
        proc withVERTEX_WEIGHT(a:shared GenSymEntry):SegGraph { components.add(Component.VERTEX_WEIGHT, a); return this; }

        proc hasSRC():bool { return components.contains(Component.SRC); }
        proc hasSRC_R():bool { return components.contains(Component.SRC_R); }
        proc hasDST():bool { return components.contains(Component.DST); }
        proc hasDST_R():bool { return components.contains(Component.DST_R); }
        proc hasSTART_IDX():bool { return components.contains(Component.START_IDX); }
        proc hasSTART_IDX_R():bool { return components.contains(Component.START_IDX_R); }
        proc hasNEIGHBOR():bool { return components.contains(Component.NEIGHBOR); }
        proc hasNEIGHBOR_R():bool { return components.contains(Component.NEIGHBOR_R); }
        proc hasEDGE_WEIGHT():bool { return components.contains(Component.EDGE_WEIGHT); }
        proc hasVERTEX_WEIGHT():bool { return components.contains(Component.VERTEX_WEIGHT); }
        
        proc getSRC() { return components.getBorrowed(Component.SRC); }
        proc getSRC_R() { return components.getBorrowed(Component.SRC_R); }
        proc getDST() { return components.getBorrowed(Component.DST); }
        proc getDST_R() { return components.getBorrowed(Component.DST_R); }
        proc getSTART_IDX() { return components.getBorrowed(Component.START_IDX); }
        proc getSTART_IDX_R() { return components.getBorrowed(Component.START_IDX_R); }
        proc getNEIGHBOR() { return components.getBorrowed(Component.NEIGHBOR); }
        proc getNEIGHBOR_R() { return components.getBorrowed(Component.NEIGHBOR_R); }
        proc getEDGE_WEIGHT() { return components.getBorrowed(Component.EDGE_WEIGHT); }
        proc getVERTEX_WEIGHT() { return components.getBorrowed(Component.VERTEX_WEIGHT); }

    }

    /**
    * GraphSymEntry is the wrapper class around SegGraph
    * so it may be stored in the Symbol Table (SymTab)
    */
    class GraphSymEntry:CompositeSymEntry {
        var dtype = NumPyDType.DType.UNDEF;
        var graph: shared SegGraph;

        proc init(segGraph: shared SegGraph) {
            super.init();
            this.entryType = SymbolEntryType.GraphSymEntry;
            assignableTypes.add(this.entryType);
            this.graph = segGraph;
        }
    }

    /**
    * Helper proc to cat AbstractSymEntry to GraphSymEntry
    */
    proc toGraphSymEntry(entry: borrowed AbstractSymEntry): borrowed GraphSymEntry throws {
        return (entry: borrowed GraphSymEntry);
    }


}
