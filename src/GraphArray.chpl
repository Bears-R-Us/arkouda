module GraphArray {

  use AryUtil;
  use CPtr;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use UnorderedCopy;
  use SipHash;
  use SegStringSort;
  use RadixSortLSD only radixSortLSD_ranks;
  use PrivateDist;
  use ServerConfig;
  use Unique;
  use Time only Timer, getCurrentTime;
  use Reflection;
  use Logging;
  use ServerErrors;
  use ArkoudaRegexCompat;

  private config const logLevel = ServerConfig.logLevel;
  const saLogger = new Logger(logLevel);

  private config param useHash = true;
  param SegmentedArrayUseHash = useHash;

  private config param regexMaxCaptures = ServerConfig.regexMaxCaptures;

  class OutOfBoundsError: Error {}




  /**
   * We use several arrays and intgers to represent a graph 
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining graph-relevant
   * operations.
   * Now we  copy from SegSArray, we need change more in the future to fit a graph
   */
  class SegGraph {
 
    /*    The starting indices for each string*/
    var n_vertices : int;

    /*    The starting indices for each string*/
    var n_edges : int;

    /*    The graph is directed (True) or undirected (False)*/
    var directed : bool;

    /*    The source of every edge in the graph, name */
    var srcName : string;

    /*    The source of every edge in the graph,array value */
    var src: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstName : string;

    /*    The destination of every vertex in the graph,array value */
    var dst: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startName : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_i: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourName : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbour : borrowed SymEntry(int);


    /*    The weitht of every vertex in the graph,name */
    var v_weightName : string;

    /*    The weitht of every vertex in the graph,array value */
    var v_weight: borrowed SymEntry(int);

    /*    The weitht of every edge in the graph, name */
    var e_weightName : string;

    /*    The weitht of every edge in the graph, array value */
    var e_weight : borrowed SymEntry(int);



    /* 
     * The following version we will init differnt kind of arrays
     * this is for src, dst, start_i, neighbour, v_weight and e_weight arrays
     */
    proc init( numv:int, nume:int, dire:bool, srcNameA: string, dstNameA: string, 
               startNameA:string,neiNameA: string, vweiNameA: string, 
               eweiNameA:string, st: borrowed SymTab) {
      n_vertices=numv;
      n_edges=nume;
      directed=dire;
      

      srcName = srcNameA;
      // The try! is needed here because init cannot throw
      var gs = try! getGenericTypedArrayEntry(srcName, st);
      var tmpsrc = toSymEntry(gs, int): unmanaged SymEntry(int);
      src=tmpsrc;

      dstName = dstNameA;
      // The try! is needed here because init cannot throw
      var ds = try! getGenericTypedArrayEntry(dstName, st);
      var tmpdst = toSymEntry(ds, int): unmanaged SymEntry(int);
      dst=tmpdst;

      startName = startNameA;
      // The try! is needed here because init cannot throw
      var starts = try! getGenericTypedArrayEntry(startName, st);
      var tmpstart_i = toSymEntry(starts, int): unmanaged SymEntry(int);
      start_i=tmpstart_i;

      neighbourName = neiNameA;
      // The try! is needed here because init cannot throw
      var neis = try! getGenericTypedArrayEntry(neighbourName, st);
      var tmpneighbour = toSymEntry(neis, int): unmanaged SymEntry(int);
      neighbour=tmpneighbour;

      v_weightName = vweiNameA;
      // The try! is needed here because init cannot throw
      var vweis = try! getGenericTypedArrayEntry(v_weightName, st);
      // I want this to be borrowed, but that throws a lifetime error
      var tmpv_weight = toSymEntry(vweis, int): unmanaged SymEntry(int);
      v_weight=tmpv_weight;

      e_weightName = eweiNameA;
      // The try! is needed here because init cannot throw
      var eweis = try! getGenericTypedArrayEntry(e_weightName, st);
      var tmpe_weight = toSymEntry(eweis, int): unmanaged SymEntry(int);
      e_weight=tmpe_weight;

    }


  } // class SegGraph




  /**
   * We use several arrays and intgers to represent a basic directed graph 
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining graph-relevant
   * operations.
   * Now we  copy from SegSArray, we need change more in the future to fit a graph
   */
  class SegGraphD {
 
    /*    The starting indices for each string*/
    var n_vertices : int;

    /*    The starting indices for each string*/
    var n_edges : int;

    /*    The graph is directed (True) or undirected (False)*/
    var directed=1 : int;

    /*    The graph is directed (True) or undirected (False)*/
    var weighted=0 : int;

    /*    The source of every edge in the graph, name */
    var srcName : string;

    /*    The source of every edge in the graph,array value */
    var src: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstName : string;

    /*    The destination of every vertex in the graph,array value */
    var dst: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startName : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_i: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourName : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbour : borrowed SymEntry(int);


    /* 
     * The following version we will init differnt kind of arrays
     * this is for src, dst, start_i, neighbour, v_weight and e_weight arrays
     */
    proc init( numv:int, nume:int, dire:int,wei:int, 
               srcNameA: string, dstNameA: string, 
               startNameA:string,neiNameA: string,  
               st: borrowed SymTab) {
      n_vertices=numv;
      n_edges=nume;
      directed=dire;
      weighted=wei;
      
      srcName = srcNameA;
      // The try! is needed here because init cannot throw
      var gs = try! getGenericTypedArrayEntry(srcName, st);
      var tmpsrc = toSymEntry(gs, int): unmanaged SymEntry(int);
      src=tmpsrc;

      dstName = dstNameA;
      // The try! is needed here because init cannot throw
      var ds = try! getGenericTypedArrayEntry(dstName, st);
      var tmpdst = toSymEntry(ds, int): unmanaged SymEntry(int);
      dst=tmpdst;

      startName = startNameA;
      // The try! is needed here because init cannot throw
      var starts = try! getGenericTypedArrayEntry(startName, st);
      var tmpstart_i = toSymEntry(starts, int): unmanaged SymEntry(int);
      start_i=tmpstart_i;

      neighbourName = neiNameA;
      // The try! is needed here because init cannot throw
      var neis = try! getGenericTypedArrayEntry(neighbourName, st);
      var tmpneighbour = toSymEntry(neis, int): unmanaged SymEntry(int);
      neighbour=tmpneighbour;
    }
  } // class SegGraphD



  /**
   * We use several arrays and intgers to represent a weighted directed graph 
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining graph-relevant
   * operations.
   * Now we  copy from SegSArray, we need change more in the future to fit a graph
   */
  class SegGraphDW {
 

    /*    The starting indices for each string*/
    var n_vertices : int;

    /*    The starting indices for each string*/
    var n_edges : int;

    /*    The graph is directed (True) or undirected (False)*/
    var directed=1 : int;

    /*    The graph is directed (True) or undirected (False)*/
    var weighted=0 : int;

    /*    The source of every edge in the graph, name */
    var srcName : string;

    /*    The source of every edge in the graph,array value */
    var src: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstName : string;

    /*    The destination of every vertex in the graph,array value */
    var dst: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startName : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_i: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourName : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbour : borrowed SymEntry(int);


    /*    The weitht of every vertex in the graph,name */
    var v_weightName : string;

    /*    The weitht of every vertex in the graph,array value */
    var v_weight: borrowed SymEntry(int);

    /*    The weitht of every edge in the graph, name */
    var e_weightName : string;

    /*    The weitht of every edge in the graph, array value */
    var e_weight : borrowed SymEntry(int);



    /* 
     * The following version we will init differnt kind of arrays
     * this is for src, dst, start_i, neighbour, v_weight and e_weight arrays
     */
    proc init( numv:int, nume:int, dire:int,wei:int, 
               srcNameA: string, dstNameA: string, 
               startNameA:string,neiNameA: string, 
               vweiNameA: string, eweiNameA:string, 
               st: borrowed SymTab) {

          n_vertices=numv;
          n_edges=nume;
          directed=dire;
          weighted=wei;
      
          srcName = srcNameA;
          // The try! is needed here because init cannot throw
          var gs = try! getGenericTypedArrayEntry(srcName, st);
          var tmpsrc = toSymEntry(gs, int): unmanaged SymEntry(int);
          src=tmpsrc;

          dstName = dstNameA;
          // The try! is needed here because init cannot throw
          var ds = try! getGenericTypedArrayEntry(dstName, st);
          var tmpdst = toSymEntry(ds, int): unmanaged SymEntry(int);
          dst=tmpdst;

          startName = startNameA;
          // The try! is needed here because init cannot throw
          var starts = try! getGenericTypedArrayEntry(startName, st);
          var tmpstart_i = toSymEntry(starts, int): unmanaged SymEntry(int);
          start_i=tmpstart_i;

          neighbourName = neiNameA;
          // The try! is needed here because init cannot throw
          var neis = try! getGenericTypedArrayEntry(neighbourName, st);
          var tmpneighbour = toSymEntry(neis, int): unmanaged SymEntry(int);
          neighbour=tmpneighbour;


          v_weightName = vweiNameA;
          // The try! is needed here because init cannot throw
          var vweis = try! getGenericTypedArrayEntry(v_weightName, st);
          var tmpv_weight = toSymEntry(vweis, int): unmanaged SymEntry(int);
          v_weight=tmpv_weight;

          e_weightName = eweiNameA;
          // The try! is needed here because init cannot throw
          var eweis = try! getGenericTypedArrayEntry(e_weightName, st);
          var tmpe_weight = toSymEntry(eweis, int): unmanaged SymEntry(int);
          e_weight=tmpe_weight;
    }
  } // class SegGraphDW




  /**
   * We use several arrays and intgers to represent an undirected graph 
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining graph-relevant
   * operations.
   * Now we  copy from SegSArray, we need change more in the future to fit a graph
   */
  class SegGraphUD {
 

    /*    The starting indices for each string*/
    var n_vertices : int;

    /*    The starting indices for each string*/
    var n_edges : int;

    /*    The graph is directed (True) or undirected (False)*/
    var directed=0 : int;

    /*    The graph is directed (True) or undirected (False)*/
    var weighted=0 : int;

    /*    The source of every edge in the graph, name */
    var srcName : string;

    /*    The source of every edge in the graph,array value */
    var src: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstName : string;

    /*    The destination of every vertex in the graph,array value */
    var dst: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startName : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_i: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourName : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbour : borrowed SymEntry(int);

    /*    The source of every edge in the graph, name */
    var srcNameR : string;

    /*    The source of every edge in the graph,array value */
    var srcR: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstNameR : string;

    /*    The destination of every vertex in the graph,array value */
    var dstR: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startNameR : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_iR: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourNameR : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbourR : borrowed SymEntry(int);


    /* 
     * The following version we will init differnt kind of arrays
     * this is for src, dst, start_i, neighbour, v_weight and e_weight arrays
     */
    proc init( numv:int, nume:int, dire:int,wei:int, 
               srcNameA: string, dstNameA: string, 
               startNameA:string,neiNameA: string, 
               srcNameAR: string, dstNameAR: string, 
               startNameAR:string,neiNameAR: string, 
               st: borrowed SymTab) {
      

          n_vertices=numv;
          n_edges=nume;
          directed=dire;
          weighted=wei;
      
          srcName = srcNameA;
          // The try! is needed here because init cannot throw
          var gs = try! getGenericTypedArrayEntry(srcName, st);
          var tmpsrc = toSymEntry(gs, int): unmanaged SymEntry(int);
          src=tmpsrc;

          dstName = dstNameA;
          // The try! is needed here because init cannot throw
          var ds = try! getGenericTypedArrayEntry(dstName, st);
          var tmpdst = toSymEntry(ds, int): unmanaged SymEntry(int);
          dst=tmpdst;

          startName = startNameA;
          // The try! is needed here because init cannot throw
          var starts = try! getGenericTypedArrayEntry(startName, st);
          var tmpstart_i = toSymEntry(starts, int): unmanaged SymEntry(int);
          start_i=tmpstart_i;

          neighbourName = neiNameA;
          // The try! is needed here because init cannot throw
          var neis = try! getGenericTypedArrayEntry(neighbourName, st);
          var tmpneighbour = toSymEntry(neis, int): unmanaged SymEntry(int);
          neighbour=tmpneighbour;



          srcNameR = srcNameAR;
          // The try! is needed here because init cannot throw
          var gsR = try! getGenericTypedArrayEntry(srcNameR, st);
          var tmpsrcR = toSymEntry(gsR, int): unmanaged SymEntry(int);
          srcR=tmpsrcR;

          dstNameR = dstNameAR;
          // The try! is needed here because init cannot throw
          var dsR = try! getGenericTypedArrayEntry(dstNameR, st);
          var tmpdstR = toSymEntry(dsR, int): unmanaged SymEntry(int);
          dstR=tmpdstR;

          startNameR = startNameAR;
          // The try! is needed here because init cannot throw
          var startsR = try! getGenericTypedArrayEntry(startNameR, st);
          var tmpstart_iR = toSymEntry(startsR, int): unmanaged SymEntry(int);
          start_iR=tmpstart_iR;

          neighbourNameR = neiNameAR;
          // The try! is needed here because init cannot throw
          var neisR = try! getGenericTypedArrayEntry(neighbourNameR, st);
          var tmpneighbourR = toSymEntry(neisR, int): unmanaged SymEntry(int);
          neighbourR=tmpneighbourR;

    }


  } // class SegGraphUD




  /**
   * We use several arrays and intgers to represent a weighted and undirected graph 
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining graph-relevant
   * operations.
   * Now we  copy from SegSArray, we need change more in the future to fit a graph
   */
  class SegGraphUDW {
 



    /*    The starting indices for each string*/
    var n_vertices : int;

    /*    The starting indices for each string*/
    var n_edges : int;

    /*    The graph is directed (True) or undirected (False)*/
    var directed=0 : int;

    /*    The graph is directed (True) or undirected (False)*/
    var weighted=0 : int;

    /*    The source of every edge in the graph, name */
    var srcName : string;

    /*    The source of every edge in the graph,array value */
    var src: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstName : string;

    /*    The destination of every vertex in the graph,array value */
    var dst: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startName : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_i: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourName : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbour : borrowed SymEntry(int);

    /*    The source of every edge in the graph, name */
    var srcNameR : string;

    /*    The source of every edge in the graph,array value */
    var srcR: borrowed SymEntry(int);

    /*    The destination of every vertex in the graph,name */
    var dstNameR : string;

    /*    The destination of every vertex in the graph,array value */
    var dstR: borrowed SymEntry(int);


    /*    The starting index  of every vertex in src and dst the ,name */
    var startNameR : string;

    /*    The starting index  of every vertex in src and dst the ,name */
    var start_iR: borrowed SymEntry(int);

    /*  The number of current vertex id v's (v<n_vertices-1) neighbours and the value is
     *  neighbour[v+1]-neighbour[v] if v<n_vertices-1 or n_edges-neighbour[v] if v=n_vertices-1
     */
    var neighbourNameR : string;

    /*    The current vertex id v's (v<n_vertices-1) neighbours are from dst[neighbour[v]] to dst[neighbour[v+1]]
     *   if v=n_vertices-1, then v's neighbours are from dst[neighbour[v]] to dst[n_edges-1], here is the value
     */
    var neighbourR : borrowed SymEntry(int);



    /*    The weitht of every vertex in the graph,name */
    var v_weightName : string;

    /*    The weitht of every vertex in the graph,array value */
    var v_weight: borrowed SymEntry(int);

    /*    The weitht of every edge in the graph, name */
    var e_weightName : string;

    /*    The weitht of every edge in the graph, array value */
    var e_weight : borrowed SymEntry(int);



    /* 
     * The following version we will init differnt kind of arrays
     * this is for src, dst, start_i, neighbour, v_weight and e_weight arrays
     */
    proc init( numv:int, nume:int, dire:int,wei:int,
               srcNameA: string, dstNameA: string, 
               startNameA:string,neiNameA: string, 
               srcNameAR: string, dstNameAR: string, 
               startNameAR:string,neiNameAR: string, 
               vweiNameA: string, eweiNameA:string, 
               st: borrowed SymTab) {


          n_vertices=numv;
          n_edges=nume;
          directed=dire;
          weighted=wei;
      
          srcName = srcNameA;
          // The try! is needed here because init cannot throw
          var gs = try! getGenericTypedArrayEntry(srcName, st);
          var tmpsrc = toSymEntry(gs, int): unmanaged SymEntry(int);
          src=tmpsrc;

          dstName = dstNameA;
          // The try! is needed here because init cannot throw
          var ds = try! getGenericTypedArrayEntry(dstName, st);
          var tmpdst = toSymEntry(ds, int): unmanaged SymEntry(int);
          dst=tmpdst;

          startName = startNameA;
          // The try! is needed here because init cannot throw
          var starts = try! getGenericTypedArrayEntry(startName, st);
          var tmpstart_i = toSymEntry(starts, int): unmanaged SymEntry(int);
          start_i=tmpstart_i;

          neighbourName = neiNameA;
          // The try! is needed here because init cannot throw
          var neis = try! getGenericTypedArrayEntry(neighbourName, st);
          var tmpneighbour = toSymEntry(neis, int): unmanaged SymEntry(int);
          neighbour=tmpneighbour;



          srcNameR = srcNameAR;
          // The try! is needed here because init cannot throw
          var gsR = try! getGenericTypedArrayEntry(srcNameR, st);
          var tmpsrcR = toSymEntry(gsR, int): unmanaged SymEntry(int);
          srcR=tmpsrcR;

          dstNameR = dstNameAR;
          // The try! is needed here because init cannot throw
          var dsR = try! getGenericTypedArrayEntry(dstNameR, st);
          var tmpdstR = toSymEntry(dsR, int): unmanaged SymEntry(int);
          dstR=tmpdstR;

          startNameR = startNameAR;
          // The try! is needed here because init cannot throw
          var startsR = try! getGenericTypedArrayEntry(startNameR, st);
          var tmpstart_iR = toSymEntry(startsR, int): unmanaged SymEntry(int);
          start_iR=tmpstart_iR;

          neighbourNameR = neiNameAR;
          // The try! is needed here because init cannot throw
          var neisR = try! getGenericTypedArrayEntry(neighbourNameR, st);
          var tmpneighbourR = toSymEntry(neisR, int): unmanaged SymEntry(int);
          neighbourR=tmpneighbourR;



          v_weightName = vweiNameA;
          // The try! is needed here because init cannot throw
          var vweis = try! getGenericTypedArrayEntry(v_weightName, st);
          // I want this to be borrowed, but that throws a lifetime error
          var tmpv_weight = toSymEntry(vweis, int): unmanaged SymEntry(int);
          v_weight=tmpv_weight;

          e_weightName = eweiNameA;
          // The try! is needed here because init cannot throw
          var eweis = try! getGenericTypedArrayEntry(e_weightName, st);
          var tmpe_weight = toSymEntry(eweis, int): unmanaged SymEntry(int);
          e_weight=tmpe_weight;

    }

  } // class SegGraphUDW









}
