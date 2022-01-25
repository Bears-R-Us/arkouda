module GraphMsg {


  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use SegmentedArray;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;


  use SymArrayDmap;
  use Random;
  use RadixSortLSD;
  use Set;
  use DistributedBag;
  use ArgSortMsg;
  use Time;
  use CommAggregation;
  use Sort;
  use Map;
  use DistributedDeque;
  use GraphArray;


  use List; 
  use LockFreeStack;
  use Atomics;
  use IO.FormattedIO; 


  private config const logLevel = ServerConfig.logLevel;
  const smLogger = new Logger(logLevel);
  
  config const start_min_degree = 1000000;
  var tmpmindegree=start_min_degree;

  private proc xlocal(x :int, low:int, high:int):bool {
      return low<=x && x<=high;
  }

  private proc xremote(x :int, low:int, high:int):bool {
      return !xlocal(x, low, high);
  }

      /* 
       * we sort the combined array [src dst] here
       */
  private proc combine_sort( src:[?D1] int, dst:[?D2] int, e_weight:[?D3] int,  weighted: bool, sortw=false: bool ) {
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=D1.size;
             //var size=Ne: int;
             var iv:[D1] int;

             //calculate how many digits is needed.
             for (bitWidth, ary, neg) in zip(bitWidths, [src,dst], negs) {
                       (bitWidth, neg) = getBitWidth(ary); 
                       totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
             }
             //sort the merged array
             proc mergedArgsort(param numDigits) throws {
                    var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
                    var curDigit = numDigits - totalDigits;
                    for (ary , nBits, neg) in zip([src,dst], bitWidths, negs) {
                        proc mergeArray(type t) {
                            ref A = ary;
                            const r = 0..#nBits by bitsPerDigit;
                            for rshift in r {
                                 const myDigit = (r.high - rshift) / bitsPerDigit;
                                 const last = myDigit == 0;
                                 forall (m, a) in zip(merged, A) {
                                     m[curDigit+myDigit] =  getDigit(a, rshift, last, neg):uint(bitsPerDigit);
                                 }
                            }
                            curDigit += r.size;
                        }
                        mergeArray(int); 
                    }
                    var tmpiv:[D1]int;
                    try {
                        tmpiv =  argsortDefault(merged);
                    } catch {
                        try! smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"error");
                    }    
                    return tmpiv;
             }

             try {
                 if totalDigits <=  2 { 
                      iv = mergedArgsort( 2); 
                 }
                 if (totalDigits >  2) && ( totalDigits <=  8) { 
                      iv =  mergedArgsort( 8); 
                 }
                 if (totalDigits >  8) && ( totalDigits <=  16) { 
                      iv = mergedArgsort(16); 
                 }
                 if (totalDigits >  16) && ( totalDigits <=  32) { 
                      iv = mergedArgsort(32); 
                 }
                 if (totalDigits >32) {    
                      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),"TotalDigits >32");
                 }

             } catch {
                  try! smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                      "error" );
             }
             var tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;
             if (weighted ){
                tmpedges=e_weight[iv];
                e_weight=tmpedges;
             }

  }//end combine_sort

      /*
       * here we preprocess the graph using reverse Cuthill.McKee algorithm to improve the locality.
       * the basic idea of RCM is relabeling the vertex based on their BFS visiting order
       */
  private proc RCM( src:[?D1] int, dst:[?D2] int, start_i:[?D3] int, neighbour:[?D4] int, depth:[?D5] int,e_weight:[?D6] int,weighted :bool )  {
          var Ne=D1.size;
          var Nv=D3.size;            
          var cmary: [0..Nv-1] int;
          var indexary:[0..Nv-1] int;
          var iv:[D1] int;
          depth=-1;
          proc smallVertex() :int {
                var minindex:int;
                for i in 0..Nv-1 {
                   if (neighbour[i]<tmpmindegree) && (neighbour[i]>0) {
                      tmpmindegree=neighbour[i];
                      minindex=i;
                   }
                }
                return minindex;
          }

          var currentindex=0:int;
          var x=smallVertex();
          cmary[0]=x;
          depth[x]=0;

          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(x);
          var numCurF=1:int;
          var GivenRatio=0.021:real;
          var topdown=0:int;
          var bottomup=0:int;
          var LF=1:int;
          var cur_level=0:int;
          
          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                //topdown, bottomup are reduce variables
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=neighbour;
                       ref sf=start_i;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];


                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           forall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) ) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                           }//end coforall
                       }else {// bottom up
                           bottomup+=1;
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.size;

                if (numCurF>0) {
                    var tmpary:[0..numCurF-1] int;
                    var sortary:[0..numCurF-1] int;
                    var numary:[0..numCurF-1] int;
                    var tmpa=0:int;
                    forall (a,b)  in zip (tmpary,SetNextF.toArray()) {
                        a=b;
                    }
                    forall i in 0..numCurF-1 {
                         numary[i]=neighbour[tmpary[i]];
                    }
                    var tmpiv:[D1]int;
                    try {
                        tmpiv =  argsortDefault(numary);
                    } catch {
                        try! smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"error");
                    }    
                    sortary=tmpary[tmpiv];
                    cmary[currentindex+1..currentindex+numCurF]=sortary;
                    currentindex=currentindex+numCurF;
                }


                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  

          if (currentindex+1<Nv) {
                 forall i in 0..Nv-1 with (+reduce currentindex) {
                     if depth[i]==-1 {
                       cmary[currentindex+1]=i;
                       currentindex+=1;  
                     }
                 }
          }
          cmary.reverse();
          forall i in 0..Nv-1{
              indexary[cmary[i]]=i;
          }

          var tmpary:[0..Ne-1] int;
          forall i in 0..Ne-1 {
                  tmpary[i]=indexary[src[i]];
          }
          src=tmpary;
          forall i in 0..Ne-1 {
                  tmpary[i]=indexary[dst[i]];
          }
          dst=tmpary;

          neighbour=0;
          start_i=-1;
          combine_sort( src, dst,e_weight,weighted, true);
          set_neighbour(src,start_i,neighbour);
  }//end RCM

  // RCM for undirected graph.
  private proc RCM_u( src:[?D1] int, dst:[?D2] int, start_i:[?D3] int, neighbour:[?D4] int, 
                      srcR:[?D5] int, dstR:[?D6] int, start_iR:[?D7] int, neighbourR:[?D8] int, 
                      depth:[?D9] int, e_weight:[?D10] int, weighted:bool )  {
              var Ne=D1.size;
              var Nv=D3.size;
              var cmary: [0..Nv-1] int;
              var indexary:[0..Nv-1] int;
              var depth:[0..Nv-1] int;
              var iv:[D1] int;
              depth=-1;
              proc smallVertex() :int {
                    var minindex:int;
                    for i in 0..Nv-1 {
                       if (neighbour[i]<tmpmindegree) && (neighbour[i]>0) {
                          tmpmindegree=neighbour[i];
                          minindex=i;
                       }
                    }
                    return minindex;
              }

              var currentindex=0:int;
              var x=smallVertex();
              cmary[0]=x;
              depth[x]=0;

              var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
              var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
              SetCurF.add(x);
              var numCurF=1:int;
              var GivenRatio=0.25:real;
              var topdown=0:int;
              var bottomup=0:int;
              var LF=1:int;
              var cur_level=0:int;
          
              while (numCurF>0) {
                    coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                       on loc {
                           ref srcf=src;
                           ref df=dst;
                           ref nf=neighbour;
                           ref sf=start_i;

                           ref srcfR=srcR;
                           ref dfR=dstR;
                           ref nfR=neighbourR;
                           ref sfR=start_iR;

                           var edgeBegin=src.localSubdomain().low;
                           var edgeEnd=src.localSubdomain().high;
                           var vertexBegin=src[edgeBegin];
                           var vertexEnd=src[edgeEnd];
                           var vertexBeginR=srcR[edgeBegin];
                           var vertexEndR=srcR[edgeEnd];

                           var switchratio=(numCurF:real)/nf.size:real;
                           if (switchratio<GivenRatio) {//top down
                               topdown+=1;
                               forall i in SetCurF with (ref SetNextF) {
                                  if ((xlocal(i,vertexBegin,vertexEnd)) ) {// current edge has the vertex
                                      var    numNF=nf[i];
                                      var    edgeId=sf[i];
                                      var nextStart=max(edgeId,edgeBegin);
                                      var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                      ref NF=df[nextStart..nextEnd];
                                      forall j in NF with (ref SetNextF){
                                             if (depth[j]==-1) {
                                                   depth[j]=cur_level+1;
                                                   SetNextF.add(j);
                                             }
                                      }
                                  } 
                                  if ((xlocal(i,vertexBeginR,vertexEndR)) )  {
                                      var    numNF=nfR[i];
                                      var    edgeId=sfR[i];
                                      var nextStart=max(edgeId,edgeBegin);
                                      var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                      ref NF=dfR[nextStart..nextEnd];
                                      forall j in NF with (ref SetNextF)  {
                                             if (depth[j]==-1) {
                                                   depth[j]=cur_level+1;
                                                   SetNextF.add(j);
                                             }
                                      }
                                  }
                               }//end coforall
                           }else {// bottom up
                               bottomup+=1;
                               forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                                  if depth[i]==-1 {
                                      var    numNF=nf[i];
                                      var    edgeId=sf[i];
                                      var nextStart=max(edgeId,edgeBegin);
                                      var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                      ref NF=df[nextStart..nextEnd];
                                      forall j in NF with (ref SetNextF){
                                             if (SetCurF.contains(j)) {
                                                   depth[i]=cur_level+1;
                                                   SetNextF.add(i);
                                             }
                                      }

                                  }
                               }
                               forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                                  if depth[i]==-1 {
                                      var    numNF=nfR[i];
                                      var    edgeId=sfR[i];
                                      var nextStart=max(edgeId,edgeBegin);
                                      var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                      ref NF=dfR[nextStart..nextEnd];
                                      forall j in NF with (ref SetNextF)  {
                                             if (SetCurF.contains(j)) {
                                                   depth[i]=cur_level+1;
                                                   SetNextF.add(i);
                                             }
                                      }
                                  }
                               }
                           }
                       }//end on loc
                    }//end coforall loc
                    cur_level+=1;
                    numCurF=SetNextF.size;

                    if (numCurF>0) {
                        var sortary:[0..numCurF-1] int;
                        var numary:[0..numCurF-1] int;
                        var tmpa=0:int;
                        var tmpary=SetNextF.toArray();
                        forall i in 0..numCurF-1 {
                             numary[i]=neighbour[tmpary[i]];
                        }
                        var tmpiv:[D1] int;
                        try {
                           tmpiv =  argsortDefault(numary);
                        } catch {
                             try! smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"error");
                        }    
                        sortary=tmpary[tmpiv];
                        cmary[currentindex+1..currentindex+numCurF]=sortary;
                        currentindex=currentindex+numCurF;
                    }


                    SetCurF=SetNextF;
                    SetNextF.clear();
              }//end while  
              if (currentindex+1<Nv) {
                 forall i in 0..Nv-1 with(+reduce currentindex) {
                     if depth[i]==-1 {
                       cmary[currentindex+1]=i;
                       currentindex+=1;  
                     }
                 }
              }
              cmary.reverse();
              forall i in 0..Nv-1{
                  indexary[cmary[i]]=i;
              }
    
              var tmpary:[0..Ne-1] int;
              forall i in 0..Ne-1 {
                      tmpary[i]=indexary[src[i]];
              }
              src=tmpary;
              forall i in 0..Ne-1 {
                      tmpary[i]=indexary[dst[i]];
              }
              dst=tmpary;
 
              neighbour=0;
              start_i=-1;
        
              combine_sort( src, dst,e_weight,weighted, true);
              set_neighbour(src,start_i,neighbour);
              coforall loc in Locales  {
                  on loc {
                      forall i in srcR.localSubdomain(){
                            srcR[i]=dst[i];
                            dstR[i]=src[i];
                       }
                  }
               }
               neighbourR=0;
               start_iR=-1;
               combine_sort( srcR, dstR,e_weight,weighted, false);
               set_neighbour(srcR,start_iR,neighbourR);
               //return true;
  }//end RCM_u

  private proc set_neighbour(src:[?D1]int, start_i :[?D2] int, neighbour :[?D3] int ){ 
          var Ne=D1.size;
          for i in 0..Ne-1 do {
             neighbour[src[i]]+=1;
             if (start_i[src[i]] ==-1){
                 start_i[src[i]]=i;
             }
          }
  }


  //sorting the vertices based on their degrees.
  private proc degree_sort(src:[?D1] int, dst:[?D2] int, start_i:[?D3] int, neighbour:[?D4] int,e_weight:[?D5] int,neighbourR:[?D6] int,weighted:bool) {
             var DegreeArray, VertexArray: [D3] int;
             var tmpedge:[D1] int;
             var Nv=D3.size;
             var iv:[D1] int;
             coforall loc in Locales  {
                on loc {
                  forall i in neighbour.localSubdomain(){
                        DegreeArray[i]=neighbour[i]+neighbourR[i];
                   }
                }
             }
             var tmpiv:[D1] int;
             try {
                 tmpiv =  argsortDefault(DegreeArray);
             } catch {
                  try! smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"error");
             }
             forall i in 0..Nv-1 {
                 VertexArray[tmpiv[i]]=i;
             }
             coforall loc in Locales  {
                on loc {
                  forall i in src.localSubdomain(){
                        tmpedge[i]=VertexArray[src[i]];
                  }
                }
             }
             src=tmpedge;
             coforall loc in Locales  {
                on loc {
                  forall i in dst.localSubdomain(){
                        tmpedge[i]=VertexArray[dst[i]];
                  }
                }
             }
             dst=tmpedge;
             coforall loc in Locales  {
                on loc {
                  forall i in src.localSubdomain(){
                        if src[i]>dst[i] {
                           src[i]<=>dst[i];
                        }
                   }
                }
             }

             combine_sort( src, dst,e_weight,weighted, true);
             neighbour=0;
             start_i=-1;
             set_neighbour(src,start_i,neighbour);

  }

  //degree sort for an undirected graph.
  private  proc degree_sort_u(src:[?D1] int, dst:[?D2] int, start_i:[?D3] int, neighbour:[?D4] int,
                      srcR:[?D5] int, dstR:[?D6] int, start_iR:[?D7] int, neighbourR:[?D8] int,e_weight:[?D9] int,weighted:bool) {

             degree_sort(src, dst, start_i, neighbour,e_weight,neighbourR,weighted);
             coforall loc in Locales  {
               on loc {
                  forall i in srcR.localSubdomain(){
                        srcR[i]=dst[i];
                        dstR[i]=src[i];
                   }
               }
             }
             combine_sort( srcR, dstR,e_weight,weighted, true);
             neighbourR=0;
             start_iR=-1;
             set_neighbour(srcR,start_iR,neighbourR);

  }

  // directly read a graph from given file and build the SegGraph class in memory
  proc segGraphFileMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var (NeS,NvS,ColS,DirectedS, FileName,RCMs,DegreeSortS) = payload.splitMsgToTuple(7);
      var Ne=NeS:int;
      var Nv=NvS:int;
      var NumCol=ColS:int;
      var directed=false:bool;
      var weighted=false:bool;
      var timer: Timer;
      var RCMFlag=false:bool;
      var DegreeSortFlag=false:bool;

      timer.start();

      if (DirectedS:int)==1 {
          directed=true;
      }
      if NumCol>2 {
           weighted=true;
      }
      if (DegreeSortS:int)==1 {
          DegreeSortFlag=true;
      }
      if (RCMs:int)==1 {
          RCMFlag=true;
      }
      var src=makeDistArray(Ne,int);
      //var edgeD:domain;
      //var vertexD:domain;
      var edgeD=src.domain;
      var neighbour=makeDistArray(Nv,int);
      var vertexD=neighbour.domain;
      var dst,e_weight,srcR,dstR, iv: [edgeD] int ;
      var start_i, depth,neighbourR, start_iR, v_weight : [vertexD] int;

      ref  ivR=iv;

      var linenum=0:int;

      var repMsg: string;

      var startpos, endpos:int;
      var sort_flag:int;
      var filesize:int;
      var tmpmindegree=start_min_degree;

      try {
           var f = open(FileName, iomode.r);
           // we check if the file can be opened correctly
           f.close();
      } catch {
                  smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      "Open file error");
      }

      proc readLinebyLine() throws {
           coforall loc in Locales  {
              on loc {
                  var f = open(FileName, iomode.r);
                  var r = f.reader(kind=ionative);
                  var line:string;
                  var a,b,c:string;
                  var curline=0:int;
                  var srclocal=src.localSubdomain();
                  var ewlocal=e_weight.localSubdomain();

                  while r.readline(line) {
                      if NumCol==2 {
                           (a,b)=  line.splitMsgToTuple(2);
                      } else {
                           (a,b,c)=  line.splitMsgToTuple(3);
                            if ewlocal.contains(curline){
                                e_weight[curline]=c:int;
                            }
                      }
                      if srclocal.contains(curline) {
                          src[curline]=(a:int)%Nv;
                          dst[curline]=(b:int)%Nv;
                      }
                      curline+=1;
                      if curline>srclocal.high {
                          break;
                      }
                  } 
                  if (curline<=srclocal.high) {
                     var outMsg="The input file " + FileName + " does not give enough edges for locale " + here.id:string;
                     smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
                  }
                  forall i in start_i.localSubdomain()  {
                       start_i[i]=-1;
                  }
                  forall i in start_iR.localSubdomain()  {
                       start_iR[i]=-1;
                  }
                  r.close();
                  f.close();
               }// end on loc
           }//end coforall
      }//end readLinebyLine
      
      // readLinebyLine sets ups src, dst, start_i, neightbor and if present e_weights so lets config them in our Graph

      readLinebyLine(); 
      timer.stop();
      

      var outMsg="Reading File takes " + timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);


      timer.clear();
      timer.start();
      combine_sort( src, dst,e_weight,weighted, true);
      set_neighbour(src,start_i,neighbour);

      // Make a composable SegGraph object that we can store in a GraphSymEntry later
      var graph = new shared SegGraph(Ne, Nv, directed);
      graph.withSRC(new shared SymEntry(src):GenSymEntry)
           .withDST(new shared SymEntry(dst):GenSymEntry)
           .withSTART_IDX(new shared SymEntry(start_i):GenSymEntry)
           .withNEIGHBOR(new shared SymEntry(neighbour):GenSymEntry);

      if (!directed) { //undirected graph
          coforall loc in Locales  {
              on loc {
                  forall i in srcR.localSubdomain(){
                        srcR[i]=dst[i];
                        dstR[i]=src[i];
                   }
              }
          }
          combine_sort( srcR, dstR,e_weight,weighted, true);
          set_neighbour(srcR,start_iR,neighbourR);

          if (DegreeSortFlag) {
             degree_sort_u(src, dst, start_i, neighbour, srcR, dstR, start_iR, neighbourR,e_weight,weighted);
          }
          if (RCMFlag) {
             RCM_u(src, dst, start_i, neighbour, srcR, dstR, start_iR, neighbourR, depth,e_weight,weighted);
          }   


          graph.withSRC_R(new shared SymEntry(srcR):GenSymEntry)
               .withDST_R(new shared SymEntry(dstR):GenSymEntry)
               .withSTART_IDX_R(new shared SymEntry(start_iR):GenSymEntry)
               .withNEIGHBOR_R(new shared SymEntry(neighbourR):GenSymEntry);

      }//end of undirected
      else {
        if (DegreeSortFlag) {
             //degree_sort(src, dst, start_i, neighbour,e_weight);
        }
        if (RCMFlag) {
             RCM(src, dst, start_i, neighbour, depth,e_weight,weighted);
        }

      }

      if (weighted) {
           graph.withEDGE_WEIGHT(new shared SymEntry(e_weight):GenSymEntry)
                .withVERTEX_WEIGHT(new shared SymEntry(v_weight):GenSymEntry);
      }

      var sNv=Nv:string;
      var sNe=Ne:string;
      var sDirected=directed:string;
      var sWeighted=weighted:string;

      var graphEntryName = st.nextName();
      var graphSymEntry = new shared GraphSymEntry(graph);
      st.addEntry(graphEntryName, graphSymEntry);
      repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted + '+created ' + graphEntryName; 
      timer.stop();
      outMsg="Sorting Edges takes "+ timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }


  //generate a graph using RMAT method.
  proc segrmatgenMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;
      var (slgNv, sNe_per_v, sp, sdire,swei,RCMs )
          = payload.splitMsgToTuple(6);

      var lgNv = slgNv: int;
      var Ne_per_v = sNe_per_v: int;
      var p = sp: real;
      //var directed=sdire : int;
      var directed=false:bool;
      //var weighted=swei : int;
      var weighted=false:bool;
      //var RCMFlag=RCMs : int;
      var RCMFlag=false:bool;
      var DegreeSortFlag=false:bool;
      var tmpmindegree=start_min_degree;

      if (sdire : int)==1 {
          directed=true;
      }
      if (swei : int)==1 {
          weighted=true;
      }
      
      if (RCMs:int)==1 {
          RCMFlag=true;
      }



      var Nv = 2**lgNv:int;
      // number of edges
      var Ne = Ne_per_v * Nv:int;

      var timer:Timer;
      timer.clear();
      timer.start();
      var n_vertices=Nv;
      var n_edges=Ne;
      var src=makeDistArray(Ne,int);
      var edgeD=src.domain;

      var neighbour=makeDistArray(Nv,int);
      var vertexD=neighbour.domain;
    

      var dst,e_weight,srcR,dstR, iv: [edgeD] int ;
      var start_i, depth,neighbourR, start_iR, v_weight : [vertexD] int;

 
      coforall loc in Locales  {
          on loc {
              forall i in src.localSubdomain() {
                  src[i]=1;
              }
              forall i in dst.localSubdomain() {
                  dst[i]=0;
              }
              forall i in start_i.localSubdomain() {
                  start_i[i]=-1;
              }
              forall i in neighbour.localSubdomain() {
                  neighbour[i]=0;
              }
          }
      }
      var srcName:string ;
      var dstName:string ;
      var startName:string ;
      var neiName:string ;
      var sNv:string;
      var sNe:string;
      var sDirected:string;
      var sWeighted:string;

      proc rmat_gen() {
             var a = p;
             var b = (1.0 - a)/ 3.0:real;
             var c = b;
             var d = b;
             var ab=a+b;
             var c_norm = c / (c + d):real;
             var a_norm = a / (a + b):real;
             // generate edges
             //var src_bit=: [0..Ne-1]int;
             //var dst_bit: [0..Ne-1]int;
             var src_bit=src;
             var dst_bit=dst;

             for ib in 1..lgNv {
                 //var tmpvar: [0..Ne-1] real;
                 var tmpvar=src;
                 fillRandom(tmpvar);
                 coforall loc in Locales  {
                       on loc {
                           forall i in src_bit.localSubdomain() {
                                 src_bit[i]=tmpvar[i]>ab;
                           }       
                       }
                 }
                 //src_bit=tmpvar>ab;
                 fillRandom(tmpvar);
                 coforall loc in Locales  {
                       on loc {
                           forall i in dst_bit.localSubdomain() {
                                 dst_bit[i]=tmpvar[i]>(c_norm * src_bit[i] + a_norm * (~ src_bit[i]));
                           }       
                       }
                 }
                 //dst_bit=tmpvar>(c_norm * src_bit + a_norm * (~ src_bit));
                 coforall loc in Locales  {
                       on loc {
                           forall i in dst.localSubdomain() {
                                 dst[i]=dst[i]+ ((2**(ib-1)) * dst_bit[i]);
                           }       
                           forall i in src.localSubdomain() {
                                 src[i]=src[i]+ ((2**(ib-1)) * src_bit[i]);
                           }       
                       }
                 }
                 //src = src + ((2**(ib-1)) * src_bit);
                 //dst = dst + ((2**(ib-1)) * dst_bit);
             }
             coforall loc in Locales  {
                       on loc {
                           forall i in src_bit.localSubdomain() {
                                 src[i]=src[i]+(src[i]==dst[i]);
                                 src[i]=src[i]%Nv;
                                 dst[i]=dst[i]%Nv;
                           }       
                       }
             }

      }//end rmat_gen
      


      rmat_gen();
      timer.stop();
      var outMsg="RMAT generate the graph takes "+timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
      timer.clear();
      timer.start();


 
      combine_sort( src, dst,e_weight,weighted, true);
      set_neighbour(src,start_i,neighbour);

      // Make a composable SegGraph object that we can store in a GraphSymEntry later
      var graph = new shared SegGraph(Ne, Nv, directed);
      graph.withSRC(new shared SymEntry(src):GenSymEntry)
               .withDST(new shared SymEntry(dst):GenSymEntry)
               .withSTART_IDX(new shared SymEntry(start_i):GenSymEntry)
               .withNEIGHBOR(new shared SymEntry(neighbour):GenSymEntry);

      if (!directed) { //undirected graph
              coforall loc in Locales  {
                  on loc {
                      forall i in srcR.localSubdomain(){
                            srcR[i]=dst[i];
                            dstR[i]=src[i];
                       }
                  }
              }
              combine_sort(srcR,dstR,e_weight, weighted, false);
              set_neighbour(srcR,start_iR,neighbourR);

              if (DegreeSortFlag) {
                 degree_sort_u(src, dst, start_i, neighbour, srcR, dstR, start_iR, neighbourR,e_weight,weighted);
              }
              if (RCMFlag) {
                 RCM_u( src, dst, start_i, neighbour, srcR, dstR, start_iR, neighbourR, depth,e_weight,weighted);
              }   

              graph.withSRC_R(new shared SymEntry(srcR):GenSymEntry)
                   .withDST_R(new shared SymEntry(dstR):GenSymEntry)
                   .withSTART_IDX_R(new shared SymEntry(start_iR):GenSymEntry)
                   .withNEIGHBOR_R(new shared SymEntry(neighbourR):GenSymEntry);
      }//end of undirected
      else {
            if (DegreeSortFlag) {
                 degree_sort(src, dst, start_i, neighbour,e_weight,neighbourR,weighted);
            }
            if (RCMFlag) {
                 RCM( src, dst, start_i, neighbour, depth,e_weight,weighted);
            }

      }//end of 
      if (weighted) {
               fillInt(e_weight,1,1000);
               fillInt(v_weight,1,1000);
               graph.withEDGE_WEIGHT(new shared SymEntry(e_weight):GenSymEntry)
                    .withVERTEX_WEIGHT(new shared SymEntry(v_weight):GenSymEntry);
      }
      var gName = st.nextName();
      st.addEntry(gName, new shared GraphSymEntry(graph));
      repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted + '+created ' + gName;

      timer.stop();
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$$$$$$ sorting RMAT graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
      outMsg="sorting RMAT graph takes "+timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);      
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  // visit a graph using BFS method
  proc segBFSMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;
      var (RCMs, n_verticesN, n_edgesN, directedN, weightedN, graphEntryName, restpart )
          = payload.splitMsgToTuple(7);
      var Nv=n_verticesN:int;
      var Ne=n_edgesN:int;
      var Directed=directedN:int;
      var Weighted=weightedN:int;
      var depthName:string;
      var RCMFlag=RCMs:int;
      var timer:Timer;



      timer.start();
      var depth=makeDistArray(Nv,int);
      coforall loc in Locales  {
                  on loc {
                           forall i in depth.localSubdomain() {
                                 depth[i]=-1;
                           }       
                  }
      }
      var root:int;
      var srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN :string;
      var srcRN, dstRN, startRN, neighbourRN:string;
      var gEntry:borrowed GraphSymEntry = getGraphSymEntry(graphEntryName, st);
      var ag = gEntry.graph;
 
      proc _d1_bfs_kernel(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int):string throws{
          var cur_level=0;
          var numCurF=1:int;//flag for stopping loop


          var edgeBeginG=makeDistArray(numLocales,int);//each locale's starting edge ID
          var edgeEndG=makeDistArray(numLocales,int);//each locales'ending edge ID

          var vertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var vertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var HvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var HvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var TvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var TvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src


          var localNum=makeDistArray(numLocales,int);// current locales's local access times
          var remoteNum=makeDistArray(numLocales,int);// current locales's remote access times
          localNum=0;
          remoteNum=0;

          var MaxBufSize=makeDistArray(numLocales,int);//temp array to calculate global max
          coforall loc in Locales   {
              on loc {
                 edgeBeginG[here.id]=src.localSubdomain().low;
                 edgeEndG[here.id]=src.localSubdomain().high;

                 vertexBeginG[here.id]=src[edgeBeginG[here.id]];
                 vertexEndG[here.id]=src[edgeEndG[here.id]];

                 if (here.id>0) {
                   HvertexBeginG[here.id]=vertexEndG[here.id-1];
                 } else {
                   HvertexBeginG[here.id]=-1;
                 }
                 if (here.id<numLocales-1) {
                   TvertexEndG[here.id]=vertexBeginG[here.id+1];
                 } else {
                   TvertexEndG[here.id]=nei.size;
                 }

                 MaxBufSize[here.id]=vertexEndG[here.id]-vertexBeginG[here.id]+1;
              }
          }
          var CMaxSize=1: int;
          for i in 0..numLocales-1 {
              if   MaxBufSize[i]> CMaxSize {
                   CMaxSize=MaxBufSize[i];
              }
          }
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          RPG=0;

          coforall loc in Locales   {
              on loc {
                 if (xlocal(root,vertexBeginG[here.id],vertexEndG[here.id]) ) {
                   localArrayG[CMaxSize*here.id]=root;
                   LPG[here.id]=1;
                   //writeln("1 Add root=",root," into locale ",here.id);
                 }
              }
          }

          while numCurF >0 {
              coforall loc in Locales   {
                  on loc {
                   ref srcf=src;
                   ref df=dst;
                   ref nf=nei;
                   ref sf=start_i;

                   //var aggdst= newDstAggregator(int);
                   //var aggsrc= newSrcAggregator(int);
                   var LocalSet= new set(int,parSafe = true);//use set to keep the next local frontier, 
                                                             //vertex in src or srcR
                   var RemoteSet=new set(int,parSafe = true);//use set to keep the next remote frontier

                   var mystart=here.id*CMaxSize;//start index 
                   //writeln("1-1 my locale=",here.id, ",has ", LPG[here.id], " elements=",localArrayG[mystart..mystart+LPG[here.id]-1],",startposition=",mystart);
                   coforall i in localArrayG[mystart..mystart+LPG[here.id]-1] with (ref LocalSet, ref RemoteSet)  {
                            // each locale just processes its local vertices
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  // i in src, current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              } 
                       
                              if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                  }
                              }//end if
                   }//end coforall
                   LPG[here.id]=0;
                   if LocalSet.size>0 {
                       LPG[here.id]=LocalSet.size;
                       localNum[here.id]+=LocalSet.size;
                       var mystart=here.id*CMaxSize;
                       forall (a,b)  in zip (localArrayG[mystart..mystart+LocalSet.size-1],LocalSet.toArray()) {
                              a=b;
                       }
                       var tmp=0;
                       for i in LocalSet {
                              tmp+=1;
                       }
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                  }//end on loc
              }//end coforall loc
              coforall loc in Locales {
                  on loc {
                   var mystart=here.id*CMaxSize;
                   for i in 0..numLocales-1 {
                       if (RPG[numLocales*i+here.id]>0) {
                           localArrayG[mystart+LPG[here.id]-1..mystart+LPG[here.id]+RPG[numLocales*i+here.id]-2]=
                               recvArrayG[CMaxSize*numLocales*i+here.id*CMaxSize..
                                          CMaxSize*numLocales*i+here.id*CMaxSize+RPG[numLocales*i+here.id]-1];
                           LPG[here.id]=LPG[here.id]+RPG[numLocales*i+here.id];
                       }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       numCurF=1;
                       break;
                   }
              }
              RPG=0;
              cur_level+=1;
          }//end while  
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          //writeln("Local Ratio=", (TotalLocal):real/(TotalLocal+TotalRemote):real,"Total Local Access=",TotalLocal," , Total Remote Access=",TotalRemote);
          return "success";
      }//end of_d1_bfs_kernel


      proc _d1_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int):string throws{
          var cur_level=0;
          var numCurF=1:int;//flag for stopping loop


          var edgeBeginG=makeDistArray(numLocales,int);//each locale's starting edge ID
          var edgeEndG=makeDistArray(numLocales,int);//each locales'ending edge ID

          var vertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var vertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var HvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var HvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var TvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var TvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src

          var vertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var vertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR
          var HvertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var HvertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR
          var TvertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var TvertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR

          var localNum=makeDistArray(numLocales,int);// current locales's local access times
          var remoteNum=makeDistArray(numLocales,int);// current locales's remote access times
          localNum=0;
          remoteNum=0;

          var MaxBufSize=makeDistArray(numLocales,int);//temp array to calculate global max
          coforall loc in Locales   {
              on loc {
                 edgeBeginG[here.id]=src.localSubdomain().low;
                 edgeEndG[here.id]=src.localSubdomain().high;

                 vertexBeginG[here.id]=src[edgeBeginG[here.id]];
                 vertexEndG[here.id]=src[edgeEndG[here.id]];

                 vertexBeginRG[here.id]=srcR[edgeBeginG[here.id]];
                 vertexEndRG[here.id]=srcR[edgeEndG[here.id]];
                 if (here.id>0) {
                   HvertexBeginG[here.id]=vertexEndG[here.id-1];
                   HvertexBeginRG[here.id]=vertexEndRG[here.id-1];
                 } else {
                   HvertexBeginG[here.id]=-1;
                   HvertexBeginRG[here.id]=-1;
                 }
                 if (here.id<numLocales-1) {
                   TvertexEndG[here.id]=vertexBeginG[here.id+1];
                   TvertexEndRG[here.id]=vertexBeginRG[here.id+1];
                 } else {
                   TvertexEndG[here.id]=nei.size;
                   TvertexEndRG[here.id]=nei.size;
                 }

                 MaxBufSize[here.id]=vertexEndG[here.id]-vertexBeginG[here.id]+
                                     vertexEndRG[here.id]-vertexBeginRG[here.id]+2;
              }
          }
          var CMaxSize=1: int;
          for i in 0..numLocales-1 {
              if   MaxBufSize[i]> CMaxSize {
                   CMaxSize=MaxBufSize[i];
              }
          }
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          RPG=0;

          coforall loc in Locales   {
              on loc {
                 if (xlocal(root,vertexBeginG[here.id],vertexEndG[here.id]) || 
                                 xlocal(root,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                   localArrayG[CMaxSize*here.id]=root;
                   LPG[here.id]=1;
                 }
              }
          }

          while numCurF >0 {
              coforall loc in Locales   {
                  on loc {
                   ref srcf=src;
                   ref df=dst;
                   ref nf=nei;
                   ref sf=start_i;

                   ref srcfR=srcR;
                   ref dfR=dstR;
                   ref nfR=neiR;
                   ref sfR=start_iR;


                   //var aggdst= newDstAggregator(int);
                   //var aggsrc= newSrcAggregator(int);
                   var LocalSet= new set(int,parSafe = true);//use set to keep the next local frontier, 
                                                             //vertex in src or srcR
                   var RemoteSet=new set(int,parSafe = true);//use set to keep the next remote frontier

                   var mystart=here.id*CMaxSize;//start index 
                   coforall i in localArrayG[mystart..mystart+LPG[here.id]-1] with (ref LocalSet, ref RemoteSet)  {
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  // i in src, current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              } 
                              if xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              }
                       
                              if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                  }
                              }//end if
                   }//end coforall
                   LPG[here.id]=0;
                   if LocalSet.size>0 {
                       LPG[here.id]=LocalSet.size;
                       localNum[here.id]+=LocalSet.size;
                       var mystart=here.id*CMaxSize;
                       forall (a,b)  in zip (localArrayG[mystart..mystart+LocalSet.size-1],LocalSet.toArray()) {
                              a=b;
                       }
                       var tmp=0;
                       for i in LocalSet {
                              tmp+=1;
                       }
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                  }//end on loc
              }//end coforall loc
              coforall loc in Locales {
                  on loc {
                   var mystart=here.id*CMaxSize;
                   for i in 0..numLocales-1 {
                       if (RPG[numLocales*i+here.id]>0) {
                           localArrayG[mystart+LPG[here.id]-1..mystart+LPG[here.id]+RPG[numLocales*i+here.id]-2]=
                               recvArrayG[CMaxSize*numLocales*i+here.id*CMaxSize..
                                          CMaxSize*numLocales*i+here.id*CMaxSize+RPG[numLocales*i+here.id]-1];
                           LPG[here.id]=LPG[here.id]+RPG[numLocales*i+here.id];
                       }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       numCurF=1;
                       break;
                   }
              }
              RPG=0;
              cur_level+=1;
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          outMsg="Local Ratio="+ ((TotalLocal):real/(TotalLocal+TotalRemote):real):string+"Total Local Access="+TotalLocal:string+ " , Total Remote Access="+ TotalRemote:string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of _d1_bfs_kernel_u

      proc fo_bag_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;
          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           forall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) || (LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) || (LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       }else {// bottom up
                           bottomup+=1;
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.getSize();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+" number of bottom up="+(bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of fo_bag_bfs_kernel_u


      proc fo_bag_bfs_kernel(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;


                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           forall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) || (LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                           }//end coforall
                       }else {// bottom up
                           bottomup+=1;
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.getSize();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of fo_bag_bfs_kernel


      proc fo_set_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           forall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) ||( LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) ||(LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       }else {//bottom up
                           bottomup+=1;
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.size;
                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);

          return "success";
      }//end of fo_set_bfs_kernel_u

      proc fo_domain_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF: domain(int);//use domain to keep the current frontier
          var SetNextF:domain(int);//use domain to keep the next frontier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           forall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) ||( LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) || (LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       } else {//bottom up
                           bottomup+=1;
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  forall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }

                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.size;
                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of fo_domain_bfs_kernel_u

      proc fo_d1_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int,GivenRatio:real):string throws{
          var cur_level=0;
          var numCurF:int=1;
          var topdown=0:int;
          var bottomup=0:int;


          var edgeBeginG=makeDistArray(numLocales,int);//each locale's starting edge ID
          var edgeEndG=makeDistArray(numLocales,int);//each locales'ending edge ID

          var vertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var vertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var HvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var TvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var BoundBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var BoundEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src

          var vertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var vertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR
          var HvertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var TvertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR

          var localNum=makeDistArray(numLocales,int);// current locales's local access times
          var remoteNum=makeDistArray(numLocales,int);// current locales's remote access times
          localNum=0;
          remoteNum=0;

          var MaxBufSize=makeDistArray(numLocales,int);//temp array to calculate global max
          coforall loc in Locales with (+ reduce topdown, + reduce bottomup)  {
              on loc {
                 edgeBeginG[here.id]=src.localSubdomain().low;
                 edgeEndG[here.id]=src.localSubdomain().high;

                 vertexBeginG[here.id]=src[edgeBeginG[here.id]];
                 vertexEndG[here.id]=src[edgeEndG[here.id]];

                 vertexBeginRG[here.id]=srcR[edgeBeginG[here.id]];
                 vertexEndRG[here.id]=srcR[edgeEndG[here.id]];

                 BoundBeginG[here.id]=vertexBeginG[here.id];
                 BoundEndG[here.id]=vertexEndG[here.id];

                 if (here.id>0) {
                   HvertexBeginG[here.id]=vertexEndG[here.id-1];
                   HvertexBeginRG[here.id]=vertexEndRG[here.id-1];
                   BoundBeginG[here.id]=min(vertexEndG[here.id-1]+1,nei.size-1);
              
                 } else {
                   HvertexBeginG[here.id]=-1;
                   HvertexBeginRG[here.id]=-1;
                   BoundBeginG[here.id]=0;
                 }
                 if (here.id<numLocales-1) {
                   TvertexEndG[here.id]=vertexBeginG[here.id+1];
                   TvertexEndRG[here.id]=vertexBeginRG[here.id+1];
                   BoundEndG[here.id]=max(BoundBeginG[here.id+1]-1,0);
                 } else {
                   TvertexEndG[here.id]=nei.size;
                   TvertexEndRG[here.id]=nei.size;
                   BoundEndG[here.id]=nei.size-1;
                 }

                 MaxBufSize[here.id]=vertexEndG[here.id]-vertexBeginG[here.id]+
                                     vertexEndRG[here.id]-vertexBeginRG[here.id]+2;
              }
          }
          var CMaxSize=1: int;
          for i in 0..numLocales-1 {
              if   MaxBufSize[i]> CMaxSize {
                   CMaxSize=MaxBufSize[i];
              }
          }
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          RPG=0;

          coforall loc in Locales   {
              on loc {
                 if (xlocal(root,vertexBeginG[here.id],vertexEndG[here.id]) || 
                                 xlocal(root,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                   localArrayG[CMaxSize*here.id]=root;
                   LPG[here.id]=1;
                 }
              }
          }

          while numCurF >0 {
              coforall loc in Locales with (+ reduce bottomup, + reduce topdown)   {
                  on loc {
                   ref srcf=src;
                   ref df=dst;
                   ref nf=nei;
                   ref sf=start_i;

                   ref srcfR=srcR;
                   ref dfR=dstR;
                   ref nfR=neiR;
                   ref sfR=start_iR;


                   var LocalSet= new set(int,parSafe = true);//use set to keep the next local frontier, 
                                                             //vertex in src or srcR
                   var RemoteSet=new set(int,parSafe = true);//use set to keep the next remote frontier

                   var mystart=here.id*CMaxSize;//start index 



                   var   switchratio=(numCurF:real)/nf.size:real;
                   if (switchratio<GivenRatio) {//top down
                       topdown+=1;
                       forall i in localArrayG[mystart..mystart+LPG[here.id]-1] 
                                                   with (ref LocalSet, ref RemoteSet)  {
                            // each locale just processes its local vertices
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  // i in src, current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  forall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              } 
                              if xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              }
                       }//end coforall
                       
                       if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                       if localeNum != here.id{
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         agg.flush();
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                       }
                                  }
                              }//end if
                   }// end of top down
                   else {  //bottom up
                       bottomup+=1;
                       proc FrontierHas(x:int):bool{
                            var returnval=false;
                            coforall i in 0..numLocales-1 with (ref returnval) {
                                if (xlocal(x,vertexBeginG[i],vertexEndG[i]) ||
                                    xlocal(x,vertexBeginRG[i],vertexEndRG[i])) {
                                    var mystart=i*CMaxSize;
                                    forall j in localArrayG[mystart..mystart+LPG[i]-1] with (ref returnval){
                                         if j==x {
                                            returnval=true;
                                         }
                                    }
                                }
                            }
                            return returnval;
                       }

                       forall i in BoundBeginG[here.id]..BoundEndG[here.id]
                                                   with (ref LocalSet, ref RemoteSet)  {
                          if (depth[i]==-1) {
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if FrontierHas(j) {
                                               depth[i]=cur_level+1;
                                               if (xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(i);
                                               } 
                                               if (xremote(i,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(i,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(i);
                                               }
                                         }
                                  }
                              } 
                              if xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if FrontierHas(j) {
                                               depth[i]=cur_level+1;
                                               if (xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(i);
                                               } 
                                               if (xremote(i,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(i,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(i);
                                               }
                                         }
                                  }
                              }
                          } //end if (depth[i]==-1)
                       
                       }//end coforall
                   }//end bottom up
                   if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                       if localeNum != here.id{
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         agg.flush();
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                       }
                                  }
                   }//end if

                   localNum[here.id]+=LPG[here.id];
                   LPG[here.id]=0;
                   if LocalSet.size>0 {
                       LPG[here.id]=LocalSet.size;
                       localNum[here.id]+=LocalSet.size;
                       var mystart=here.id*CMaxSize;
                       forall (a,b)  in zip (localArrayG[mystart..mystart+LocalSet.size-1],LocalSet.toArray()) {
                              a=b;
                       }
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                  }//end on loc
              }//end coforall loc
              coforall loc in Locales {
                  on loc {
                   var mystart=here.id*CMaxSize;
                   for i in 0..numLocales-1 {
                     if i != here.id {
                       if (RPG[here.id*numLocales+i]>0) {
                           localArrayG[mystart+LPG[here.id]..mystart+LPG[here.id]+RPG[numLocales*here.id+i]-1]=
                           recvArrayG[CMaxSize*numLocales*here.id+i*CMaxSize..
                                      CMaxSize*numLocales*here.id+i*CMaxSize+RPG[numLocales*here.id+i]-1];
                           LPG[here.id]=LPG[here.id]+RPG[numLocales*here.id+i];
                       }
                     }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       numCurF+=LPG[iL];
                   }
              }
              RPG=0;
              cur_level+=1;
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          outMsg="Local Ratio="+ ((TotalLocal):real/(TotalLocal+TotalRemote):real):string +"Total Local Access="+ (TotalLocal:string) +" , Total Remote Access="+ (TotalRemote:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of fo_d1_bfs_kernel_u

      proc co_bag_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          SetCurF.add(root);
          var numCurF=1:int;
          var bottomup=0:int;
          var topdown=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           coforall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) || (LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) || (LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       }else {// bottom up
                           bottomup+=1;

                           coforall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.getSize();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of co_bag_bfs_kernel_u


      proc co_set_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;

          var bottomup=0:int;
          var topdown=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           coforall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) ||( LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) ||(LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       }else {//bottom up
                           bottomup+=1;
                           coforall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }
                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.size;
                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of co_set_bfs_kernel_u

      proc co_domain_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF: domain(int);//use domain to keep the current frontier
          var SetNextF:domain(int);//use domain to keep the next frontier
          SetCurF.add(root);
          var numCurF=1:int;
          var bottomup=0:int;
          var topdown=0:int;

          while (numCurF>0) {
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=nei;
                       ref sf=start_i;

                       ref srcfR=srcR;
                       ref dfR=dstR;
                       ref nfR=neiR;
                       ref sfR=start_iR;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];
                       var vertexBeginR=srcR[edgeBegin];
                       var vertexEndR=srcR[edgeEnd];

                       var switchratio=(numCurF:real)/nf.size:real;
                       if (switchratio<GivenRatio) {//top down
                           topdown+=1;
                           coforall i in SetCurF with (ref SetNextF) {
                              if ((xlocal(i,vertexBegin,vertexEnd)) ||( LF==0)) {// current edge has the vertex
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              } 
                              if ((xlocal(i,vertexBeginR,vertexEndR)) || (LF==0))  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               SetNextF.add(j);
                                         }
                                  }
                              }
                           }//end coforall
                       } else {//bottom up
                           bottomup+=1;
                           coforall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF){
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }

                              }
                           }
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                              if depth[i]==-1 {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBegin);
                                  var nextEnd=min(edgeEnd,edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref SetNextF)  {
                                         if (SetCurF.contains(j)) {
                                               depth[i]=cur_level+1;
                                               SetNextF.add(i);
                                         }
                                  }
                              }
                           }

                       }
                   }//end on loc
                }//end coforall loc
                cur_level+=1;
                numCurF=SetNextF.size;
                SetCurF=SetNextF;
                SetNextF.clear();
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of co_domain_bfs_kernel_u

      proc co_d1_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int,GivenRatio:real):string throws{
          var cur_level=0;
          var numCurF=1:int;//flag for stopping loop
          var topdown=0:int;
          var bottomup=0:int;


          var edgeBeginG=makeDistArray(numLocales,int);//each locale's starting edge ID
          var edgeEndG=makeDistArray(numLocales,int);//each locales'ending edge ID

          var vertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var vertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var HvertexBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var TvertexEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src
          var BoundBeginG=makeDistArray(numLocales,int);//each locale's beginning vertex ID in src
          var BoundEndG=makeDistArray(numLocales,int);// each locales' ending vertex ID in src

          var vertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var vertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR
          var HvertexBeginRG=makeDistArray(numLocales,int);// each locales' beginning vertex ID in srcR
          var TvertexEndRG=makeDistArray(numLocales,int);// each locales's ending vertex ID in srcR

          var localNum=makeDistArray(numLocales,int);// current locales's local access times
          var remoteNum=makeDistArray(numLocales,int);// current locales's remote access times
          localNum=0;
          remoteNum=0;

          var MaxBufSize=makeDistArray(numLocales,int);//temp array to calculate global max
          coforall loc in Locales   {
              on loc {
                 edgeBeginG[here.id]=src.localSubdomain().low;
                 edgeEndG[here.id]=src.localSubdomain().high;

                 vertexBeginG[here.id]=src[edgeBeginG[here.id]];
                 vertexEndG[here.id]=src[edgeEndG[here.id]];

                 vertexBeginRG[here.id]=srcR[edgeBeginG[here.id]];
                 vertexEndRG[here.id]=srcR[edgeEndG[here.id]];

                 BoundBeginG[here.id]=vertexBeginG[here.id];
                 BoundEndG[here.id]=vertexEndG[here.id];

                 if (here.id>0) {
                   HvertexBeginG[here.id]=vertexEndG[here.id-1];
                   HvertexBeginRG[here.id]=vertexEndRG[here.id-1];
                   BoundBeginG[here.id]=min(vertexEndG[here.id-1]+1,nei.size-1);
              
                 } else {
                   HvertexBeginG[here.id]=-1;
                   HvertexBeginRG[here.id]=-1;
                   BoundBeginG[here.id]=0;
                 }
                 if (here.id<numLocales-1) {
                   TvertexEndG[here.id]=vertexBeginG[here.id+1];
                   TvertexEndRG[here.id]=vertexBeginRG[here.id+1];
                   BoundEndG[here.id]=max(BoundBeginG[here.id+1]-1,0);
                 } else {
                   TvertexEndG[here.id]=nei.size;
                   TvertexEndRG[here.id]=nei.size;
                   BoundEndG[here.id]=nei.size-1;
                 }

                 MaxBufSize[here.id]=vertexEndG[here.id]-vertexBeginG[here.id]+
                                     vertexEndRG[here.id]-vertexBeginRG[here.id]+2;
              }
          }
          var CMaxSize=1: int;
          for i in 0..numLocales-1 {
              if   MaxBufSize[i]> CMaxSize {
                   CMaxSize=MaxBufSize[i];
              }
          }
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          RPG=0;

          coforall loc in Locales   {
              on loc {
                 if (xlocal(root,vertexBeginG[here.id],vertexEndG[here.id]) || 
                                 xlocal(root,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                   localArrayG[CMaxSize*here.id]=root;
                   LPG[here.id]=1;
                 }
              }
          }

          while numCurF >0 {
              coforall loc in Locales with (+ reduce topdown, + reduce bottomup)  {
                  on loc {
                   ref srcf=src;
                   ref df=dst;
                   ref nf=nei;
                   ref sf=start_i;

                   ref srcfR=srcR;
                   ref dfR=dstR;
                   ref nfR=neiR;
                   ref sfR=start_iR;


                   var LocalSet= new set(int,parSafe = true);//use set to keep the next local frontier, 
                                                             //vertex in src or srcR
                   var RemoteSet=new set(int,parSafe = true);//use set to keep the next remote frontier

                   var mystart=here.id*CMaxSize;//start index 



                   var   switchratio=(numCurF:real)/nf.size:real;
                   if (switchratio<GivenRatio) {//top down
                       topdown+=1;
                       coforall i in localArrayG[mystart..mystart+LPG[here.id]-1] 
                                                   with (ref LocalSet, ref RemoteSet)  {
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              } 
                              if xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if (depth[j]==-1) {
                                               depth[j]=cur_level+1;
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(j);
                                               }
                                         }
                                  }
                              }
                       
                       }//end coforall
                       if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                       if localeNum != here.id{
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         agg.flush();
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                       }
                                  }
                       }//end if
                   }// end of top down
                   else {  //bottom up
                       bottomup+=1;
                       proc FrontierHas(x:int):bool{
                            var returnval=false;
                            coforall i in 0..numLocales-1 with (ref returnval) {
                                if (xlocal(x,vertexBeginG[i],vertexEndG[i]) ||
                                    xlocal(x,vertexBeginRG[i],vertexEndRG[i])) {
                                    var mystart=i*CMaxSize;
                                    for j in localArrayG[mystart..mystart+LPG[i]-1] {
                                         if j==x {
                                            returnval=true;
                                            break;
                                         }
                                    }
                                }
                            }
                            return returnval;
                       }

                       coforall i in BoundBeginG[here.id]..BoundEndG[here.id]
                                                   with (ref LocalSet, ref RemoteSet)  {
                          if (depth[i]==-1) {
                              if xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) {
                                  var    numNF=nf[i];
                                  var    edgeId=sf[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=df[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if FrontierHas(j) {
                                               depth[i]=cur_level+1;
                                               if (xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(i);
                                               } 
                                               if (xremote(i,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(i,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(i);
                                               }
                                         }
                                  }
                              } 
                              if xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])  {
                                  var    numNF=nfR[i];
                                  var    edgeId=sfR[i];
                                  var nextStart=max(edgeId,edgeBeginG[here.id]);
                                  var nextEnd=min(edgeEndG[here.id],edgeId+numNF-1);
                                  ref NF=dfR[nextStart..nextEnd];
                                  coforall j in NF with (ref LocalSet, ref RemoteSet) {
                                         if FrontierHas(j) {
                                               depth[i]=cur_level+1;
                                               if (xlocal(i,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(i,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    LocalSet.add(i);
                                               } 
                                               if (xremote(i,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(i,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    RemoteSet.add(i);
                                               }
                                         }
                                  }
                              }
                          } //end if (depth[i]==-1)
                       }//end coforall
                   }// end bottom up
                       
                   if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  coforall localeNum in 0..numLocales-1  { 
                                       if localeNum != here.id{
                                         var ind=0:int;
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                }
                                         }
                                         agg.flush();
                                         RPG[localeNum*numLocales+here.id]=ind;
                                         ind=0;
                                       }
                                  }
                   }//end if

                   localNum[here.id]+=LPG[here.id];
                   LPG[here.id]=0;
                   if LocalSet.size>0 {
                       LPG[here.id]=LocalSet.size;
                       localNum[here.id]+=LocalSet.size;
                       var mystart=here.id*CMaxSize;
                       forall (a,b)  in zip (localArrayG[mystart..mystart+LocalSet.size-1],LocalSet.toArray()) {
                              a=b;
                       }
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                  }//end on loc
              }//end coforall loc
              coforall loc in Locales {
                  on loc {
                   var mystart=here.id*CMaxSize;
                   for i in 0..numLocales-1 {
                     if i != here.id {
                       if (RPG[here.id*numLocales+i]>0) {
                           localArrayG[mystart+LPG[here.id]..mystart+LPG[here.id]+RPG[numLocales*here.id+i]-1]=
                           recvArrayG[CMaxSize*numLocales*here.id+i*CMaxSize..
                                      CMaxSize*numLocales*here.id+i*CMaxSize+RPG[numLocales*here.id+i]-1];
                           LPG[here.id]=LPG[here.id]+RPG[numLocales*here.id+i];
                       }
                     }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       numCurF+=LPG[iL];
                       //break;
                   }
              }
              RPG=0;
              cur_level+=1;
          }//end while  
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg="number of top down = "+(topdown:string)+ " number of bottom up="+ (bottomup:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          outMsg="Local Ratio="+ ((TotalLocal):real/(TotalLocal+TotalRemote):real):string + "Total Local Access=" + (TotalLocal:string) +" Total Remote Access=" + (TotalRemote:string);
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          return "success";
      }//end of co_d1_bfs_kernel_u

      proc return_depth(): string throws{
          var depthName = st.nextName();
          var depthEntry = new shared SymEntry(depth);
          st.addEntry(depthName, depthEntry);
          //try! st.addEntry(vertexName, vertexEntry);

          var depMsg =  'created ' + st.attrib(depthName);
          //var lrepMsg =  'created ' + st.attrib(levelName) + '+created ' + st.attrib(vertexName) ;
          return depMsg;

      }

      if (Directed!=0) {
          if (Weighted!=0) {
              var ratios:string;
               (srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN,ratios)=
                   restpart.splitMsgToTuple(8);
              root=rootN:int;
              var GivenRatio=ratios:real;
              if (RCMFlag>0) {
                  root=0;
              }
              depth[root]=0;


            //   var ag = new owned SegGraphDW(Nv,Ne,Directed,Weighted,srcN,dstN,
            //                      startN,neighbourN,vweightN,eweightN, st);
            //   fo_bag_bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,1,GivenRatio);
              fo_bag_bfs_kernel(
                  toSymEntry(ag.getNEIGHBOR(), int).a,
                  toSymEntry(ag.getSTART_IDX(), int).a,
                  toSymEntry(ag.getSRC(), int).a,
                  toSymEntry(ag.getDST(), int).a,
                  1, GivenRatio);

               repMsg=return_depth();

          } else {
              var ratios:string;

              (srcN, dstN, startN, neighbourN,rootN,ratios )=restpart.splitMsgToTuple(6);
              root=rootN:int;
              var GivenRatio=ratios:real;
              if (RCMFlag>0) {
                  root=0;
              }
              depth[root]=0;
             //   fo_bag_bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,1,GivenRatio);
              fo_bag_bfs_kernel(
                  toSymEntry(ag.getNEIGHBOR(), int).a,
                  toSymEntry(ag.getSTART_IDX(), int).a,
                  toSymEntry(ag.getSRC(), int).a,
                  toSymEntry(ag.getDST(), int).a,
                  1, GivenRatio);

               repMsg=return_depth();
          }
      }
      else {
          if (Weighted!=0) {
              var ratios:string;
              (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN,vweightN,eweightN, rootN, ratios)=
                   restpart.splitMsgToTuple(12);
              root=rootN:int;
              if (RCMFlag>0) {
                  root=0;
              }
              depth=-1;
              depth[root]=0;
              var Flag=0:int;
              var GivenRatio=ratios:real;
              if (GivenRatio <0  ) {//do default call
                  GivenRatio=-1.0* GivenRatio;
                  //co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                  //         ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                 //   fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                //            ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,1,GivenRatio);
                  fo_bag_bfs_kernel_u(
                      toSymEntry(ag.getNEIGHBOR(), int).a,
                      toSymEntry(ag.getSTART_IDX(), int).a,
                      toSymEntry(ag.getSRC(), int).a,
                      toSymEntry(ag.getDST(), int).a,
                      toSymEntry(ag.getNEIGHBOR_R(), int).a,
                      toSymEntry(ag.getSTART_IDX_R(), int).a,
                      toSymEntry(ag.getSRC_R(), int).a,
                      toSymEntry(ag.getDST_R(), int).a,
                      1, GivenRatio
                  );
                  
                  repMsg=return_depth();
 
              } else {// do batch test
                  depth=-1;
                  depth[root]=0;
                  timer.stop();
                  timer.clear();
                  timer.start();
 
                //   co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                //            ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                  co_d1_bfs_kernel_u(
                      toSymEntry(ag.getNEIGHBOR(), int).a,
                      toSymEntry(ag.getSTART_IDX(), int).a,
                      toSymEntry(ag.getSRC(), int).a,
                      toSymEntry(ag.getDST(), int).a,
                      toSymEntry(ag.getNEIGHBOR_R(), int).a,
                      toSymEntry(ag.getSTART_IDX_R(), int).a,
                      toSymEntry(ag.getSRC_R(), int).a,
                      toSymEntry(ag.getDST_R(), int).a,
                      GivenRatio
                  );
                  timer.stop();
                  var outMsg= "graph BFS takes "+timer.elapsed():string+ " for Co D Hybrid version";
                  smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);

                  /*
                  depth=-1;
                  depth[root]=0;
                  Flag=1;
                  timer.clear();
                  timer.start();
                  co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,2.0);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co D TopDown version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Bag L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Bag G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Set L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Set G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Domain L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Domain G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  fo_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for D Hybrid version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  fo_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,2.0);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for D TopDown version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Bag L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Bag G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Set L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  fo_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Set G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Domain L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  fo_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Domain G version $$$$$$$$$$$$$$$$$$");
                  */

                  repMsg=return_depth();
              }//end of batch test

          } else {
              var ratios:string;
              (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN, rootN,ratios )=
                   restpart.splitMsgToTuple(10);
 
              root=rootN:int;
              if (RCMFlag>0) {
                  root=0;
              }
              depth=-1;
              depth[root]=0;
              var Flag=0:int;
              var GivenRatio=ratios:real;
              if (GivenRatio <0 ) {//do default call
                  GivenRatio=-1.0*GivenRatio;
                  //co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                  //         ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                   fo_bag_bfs_kernel_u(
                      toSymEntry(ag.getNEIGHBOR(), int).a,
                      toSymEntry(ag.getSTART_IDX(), int).a,
                      toSymEntry(ag.getSRC(), int).a,
                      toSymEntry(ag.getDST(), int).a,
                      toSymEntry(ag.getNEIGHBOR_R(), int).a,
                      toSymEntry(ag.getSTART_IDX_R(), int).a,
                      toSymEntry(ag.getSRC_R(), int).a,
                      toSymEntry(ag.getDST_R(), int).a,
                      1, GivenRatio);
                  
                   repMsg=return_depth();
 
              } else {// do batch test
                  timer.stop();
                  timer.clear();
                  timer.start();
 
                  co_d1_bfs_kernel_u(
                      toSymEntry(ag.getNEIGHBOR(), int).a,
                      toSymEntry(ag.getSTART_IDX(), int).a,
                      toSymEntry(ag.getSRC(), int).a,
                      toSymEntry(ag.getDST(), int).a,
                      toSymEntry(ag.getNEIGHBOR_R(), int).a,
                      toSymEntry(ag.getSTART_IDX_R(), int).a,
                      toSymEntry(ag.getSRC_R(), int).a,
                      toSymEntry(ag.getDST_R(), int).a,
                      GivenRatio);
                  timer.stop();
                  var outMsg= "graph BFS takes "+timer.elapsed():string+ " for Co D Hybrid version";
                  smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
                  /*
                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,2.0);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co D TopDown version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Bag L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Bag G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Set L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Set G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  co_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Domain L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  co_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co Domain G version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  fo_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for D Hybrid version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  fo_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,2.0);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for D TopDown version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Bag L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.start();
                  Flag=0;
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Bag G version $$$$$$$$$$$$$$$$$$");

 
                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Set L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  fo_set_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Set G version $$$$$$$$$$$$$$$$$$");


                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=1;
                  fo_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Domain L version $$$$$$$$$$$$$$$$$$");

                  depth=-1;
                  depth[root]=0;
                  timer.clear();
                  timer.start();
                  Flag=0;
                  fo_domain_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,Flag,GivenRatio);
                  timer.stop();
                  writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Domain G version $$$$$$$$$$$$$$$$$$");

                  */

                  repMsg=return_depth();
              }
          }
      }
      timer.stop();
      //writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
      var outMsg= "graph BFS takes "+timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc registerMe() {
        use CommandMap;
        registerFunction("segmentedGraphFile", segGraphFileMsg);
        registerFunction("segmentedRMAT", segrmatgenMsg);
        registerFunction("segmentedGraphBFS", segBFSMsg);
    }
 }


