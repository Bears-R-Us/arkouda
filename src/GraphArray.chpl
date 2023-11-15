module GraphArray {
    // Chapel modules.
    use Map;
    use Reflection;
    use Utils;
    
    // Arkouda modules.
    use Logging;
    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    // Server message logger.
    private config const logLevel = LogLevel.DEBUG;
    const graphLogger = new Logger(logLevel);

    // Component key names to be stored stored in the components map for future retrieval
    enum Component {
        SRC,                    // The source array of every edge in the graph
        DST,                    // The destination array of every edge in the graph
        SEGMENTS,               // The segments of adjacency lists for each vertex in DST
        RANGES,                 // Keeps the range of the vertices the edge list stores per locale
        EDGE_WEIGHT,            // Stores the edge weights of the graph, if applicable
        NODE_MAP,               // Doing an index of NODE_MAP[u] gives you the original value of u
        VERTEX_LABELS,          // Any labels that belong to a specific node
        VERTEX_LABELS_MAP,      // Sorted array of vertex labels to integer id (array index)
        EDGE_RELATIONSHIPS,     // The relationships that belong to specific edges
        EDGE_RELATIONSHIPS_MAP, // Sorted array of edge relationships to integer id (array index)
        VERTEX_PROPS,           // Any properties that belong to a specific node
        VERTEX_PROPS_COL_MAP,   // Sorted array of vertex property to integer id (array index)
        VERTEX_PROPS_DTYPE_MAP, // Sorted array of column datatype to integer id (array index)
        VERTEX_PROPS_COL2DTYPE, // Map of column names to the datatype of the column 
        EDGE_PROPS,             // Any properties that belong to a specific edge
        EDGE_PROPS_COL_MAP,     // Sorted array of edge property to integer id (array index)
        EDGE_PROPS_DTYPE_MAP,   // Sorted array of column datatype to integer id (array index)
        EDGE_PROPS_COL2DTYPE,   // Map of column names to the datatype of the column

        // TEMPORARY COMPONENTS BELOW FOR UNDIRECTED GRAPHS TO ENSURE COMPATIBILTIY WITH OLD CODE.
        // We want to phase out the need for reversed arrays in undirected graph algorithms.
        // Includes: connected components, triangle counting, k-truss, and triangle centrality.
        SRC_R,                  // DST array
        DST_R,                  // SRC array
        START_IDX,              // Starting indices of vertices in SRC
        START_IDX_R,            // Starting indices of vertices in SRC_R
        NEIGHBOR,               // Number of neighbors for a given vertex based off SRC and DST
        NEIGHBOR_R,             // Number of neighbors for a given vertex based off SRC_R and DST_R
    }

    /**
    * We use several arrays and integers to represent a graph.
    * Instances are ephemeral, not stored in the symbol table. Instead, attributes
    * of this class refer to symbol table entries that persist. This class is a
    * convenience for bundling those persistent objects and defining graph-relevant
    * operations.
    */
    class SegGraph {
        // Map to the hold various components of our Graph; use enum Component values as map keys
        var components = new map(Component, shared GenSymEntry, parSafe=false);

        // Total number of vertices
        var n_vertices : int;

        // Total number of edges
        var n_edges : int;

        // The graph is directed (true) or undirected (false)
        var directed : bool;

        // The graph is weighted (true) or unweighted (false)
        var weighted: bool;

        // The graph is a property graph (true) or not (false)
        var propertied: bool;

        // Undirected graphs are in the old format (true) or not (false)
        var reversed: bool = false;

        /**
        * Init the basic graph object, we'll compose the pieces using the withComp method.
        */
        proc init(num_v:int, num_e:int, directed:bool, weighted:bool, propertied:bool) {
            this.n_vertices = num_v;
            this.n_edges = num_e;
            this.directed = directed;
            this.weighted = weighted;
            this.propertied = propertied;
        }

        proc isDirected():bool { return this.directed; }
        proc isWeighted():bool { return this.weighted; }
        proc isPropertied():bool { return this.propertied; }

        proc withComp(a:shared GenSymEntry, atrname:string):SegGraph throws { components.add(atrname:Component, a); return this; }
        proc hasComp(atrname:string):bool throws { return components.contains(atrname:Component); }
        proc getComp(atrname:string):GenSymEntry throws { return components[atrname:Component]; }
    }

    /**
    * GraphSymEntry is the wrapper class around SegGraph so it may be stored in 
    * the Symbol Table (SymTab).
    */
    class GraphSymEntry : CompositeSymEntry {
        var graph: shared SegGraph;

        proc init(segGraph: shared SegGraph) {
            super.init();
            this.entryType = SymbolEntryType.CompositeSymEntry;
            assignableTypes.add(this.entryType);
            this.graph = segGraph;
        }
        override proc getSizeEstimate(): int {
            return 1;
        }
    }

    class SymEntryAD : GenSymEntry {
        var aD: domain(int);
        var a: [aD] int;
        
        proc init(associative_array: [?associative_domain] int) {
            super.init(int);
            this.aD = associative_domain;
            this.a = associative_array;
        }
    }

    class MapSymEntry : GenSymEntry {
        var stored_map: map(string, string);
        
        proc init(ref map_to_store: map(string, string)) {
            super.init(string);
            this.stored_map = map_to_store;
        }
    }

    proc toMapSymEntry(e) {
        return try! e : borrowed MapSymEntry;
    }

    proc toSymEntryAD(e) {
        return try! e : borrowed SymEntryAD();
    }

    /**
    * Convenience proc to retrieve GraphSymEntry from SymTab.
    * Performs conversion from AbstractySymEntry to GraphSymEntry.
    */
    proc getGraphSymEntry(name:string, st: borrowed SymTab): borrowed GraphSymEntry throws {
        var abstractEntry:borrowed AbstractSymEntry = st.lookup(name);
        if !abstractEntry.isAssignableTo(SymbolEntryType.CompositeSymEntry) {
            var errorMsg = "Error: SymbolEntryType %s is not assignable to CompositeSymEntry".format(abstractEntry.entryType);
            graphLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new Error(errorMsg);
        }
        return (abstractEntry: borrowed GraphSymEntry);
    }

    /**
    * Helper proc to cast AbstractSymEntry to GraphSymEntry.
    */
    proc toGraphSymEntry(entry: borrowed AbstractSymEntry): borrowed GraphSymEntry throws {
        return (entry: borrowed GraphSymEntry);
    }
}