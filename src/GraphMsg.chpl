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


  use GenSymIO only jsonToPdArray,jsonToPdArrayInt;
  use SymArrayDmap;
  use Random;
  use RadixSortLSD;
  use Set;
  use DistributedBag;
  public use ArgSortMsg;
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
  


  // directly read a graph from given file and build the SegGraph class in memory
  proc segGraphFileMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var (NeS,NvS,ColS,DirectedS, FileName,RCMs,DegreeSortS) = payload.splitMsgToTuple(7);
      //writeln("======================Graph Reading=====================");
      var Ne=NeS:int;
      var Nv=NvS:int;
      var NumCol=ColS:int;
      var directed=DirectedS:int;
      var weighted=0:int;
      var timer: Timer;
      var RCMFlag=RCMs:int;
      var DegreeSortFlag=DegreeSortS:int;
      if NumCol>2 {
           weighted=1;
      }

      timer.start();
      var src=makeDistArray(Ne,int);
      var dst=makeDistArray(Ne,int);
      var neighbour=makeDistArray(Nv,int);
      var start_i=makeDistArray(Nv,int);
      var depth=makeDistArray(Nv,int);

      var e_weight = makeDistArray(Ne,int);
      var v_weight = makeDistArray(Nv,int);

      var iv=makeDistArray(Ne,int);

      var srcR=makeDistArray(Ne,int);
      var dstR=makeDistArray(Ne,int);
      var neighbourR=makeDistArray(Nv,int);
      var start_iR=makeDistArray(Nv,int);
      ref  ivR=iv;

      var linenum=0:int;

      var repMsg: string;

      var startpos, endpos:int;
      var sort_flag:int;
      var filesize:int;

      proc readLinebyLine() throws {
           coforall loc in Locales  {
              on loc {
                  var f = open(FileName, iomode.r);
                  var r = f.reader(kind=ionative);
                  var line:string;
                  var a,b,c:string;
                  var curline=0:int;
                  var srclocal=src.localSubdomain();
                  var dstlocal=dst.localSubdomain();
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
                          src[curline]=a:int;
                          dst[curline]=b:int;
                      }
                      //if dstlocal.contains(curline) {
                      //    dst[curline]=b:int;
                      //}
                      curline+=1;
                      if curline>srclocal.high {
                          break;
                      }
                  } 
                  if (curline<=srclocal.high) {
                     //writeln("XXXXXXXXXXXXXXXXXXXXXXXXXXX");
                     //writeln("The input file ",FileName, " does not give enough edges for locale ", here.id);
                     var outMsg="The input file " + FileName + " does not give enough edges for locale " + here.id:string;
                     smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
                     //writeln("XXXXXXXXXXXXXXXXXXXXXXXXXXX");
                  }
                  forall i in src.localSubdomain() {
                  //     src[i]=src[i]+(src[i]==dst[i]);
                       src[i]=src[i]%Nv;
                       dst[i]=dst[i]%Nv;
                  }
                  forall i in start_i.localSubdomain()  {
                       start_i[i]=-1;
                  }
                  forall i in neighbour.localSubdomain()  {
                       neighbour[i]=0;
                  }
                  forall i in start_iR.localSubdomain()  {
                       start_iR[i]=-1;
                  }
                  forall i in neighbourR.localSubdomain()  {
                       neighbourR[i]=0;
                  }
                  r.close();
                  f.close();
               }// end on loc
           }//end coforall
      }//end readLinebyLine
      

      proc combine_sort() throws {
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;

             for (bitWidth, ary, neg) in zip(bitWidths, [src,dst], negs) {
                       (bitWidth, neg) = getBitWidth(ary); 
                       totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
             }
             proc mergedArgsort(param numDigits) throws {
                    //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
                    //             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
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
                    var tmpiv = argsortDefault(merged);
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
                      return "Error, TotalDigits >32";
                 }

             } catch e: Error {
                  smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                    return "Error: %t".format(e.message());
             }
             var tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;
             if (weighted){
                tmpedges=e_weight[iv];
                e_weight=tmpedges;
             }

             return "success";
      }//end combine_sort


      proc RCM() throws {
            
          var cmary: [0..Nv-1] int;
          var indexary:[0..Nv-1] int;
          depth=-1;
          proc smallVertex() :int {
                var tmpmindegree=1000000:int;
                var minindex=0:int;
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

          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(x);
          var numCurF=1:int;
          var GivenRatio=0.021:int;
          var topdown=0:int;
          var bottomup=0:int;
          var LF=1:int;
          var cur_level=0:int;
          
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=neighbour;
                       ref sf=start_i;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }
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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
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

                    var tmpiv = argsortDefault(numary);
                    sortary=tmpary[tmpiv];
                    cmary[currentindex+1..currentindex+numCurF]=sortary;
                    currentindex=currentindex+numCurF;
                }


                //SetCurF<=>SetNextF;
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

          return "success";
      }//end RCM

      proc set_neighbour(){ 
          for i in 0..Ne-1 do {
             neighbour[src[i]]+=1;
             if (start_i[src[i]] ==-1){
                 start_i[src[i]]=i;
             }
          }
      }

      readLinebyLine();
      timer.stop();

      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$ Reading File takes ", timer.elapsed()," $$$$$$$$$$$$$$$$$$$$$$$");

      var outMsg="Reading File takes " + timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");

      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");

      timer.clear();
      timer.start();
      combine_sort();
      set_neighbour();

      if (directed==0) { //undirected graph

          proc combine_sortR() throws {
             /* we cannot use the coargsort version because it will break the memory limit */
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;
             for (bitWidth, ary, neg) in zip(bitWidths, [srcR,dstR], negs) {
                 (bitWidth, neg) = getBitWidth(ary); 
                 totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;

             }
             proc mergedArgsort(param numDigits) throws {
               //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
               //          + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
               var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
               var curDigit = numDigits - totalDigits;
               for (ary , nBits, neg) in zip([srcR,dstR], bitWidths, negs) {
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
               var tmpiv = argsortDefault(merged);
               return tmpiv;
             } 

             try {
                 if totalDigits <=  2 { 
                      ivR = mergedArgsort( 2); 
                 }
                 if (totalDigits >  2) && ( totalDigits <=  8) { 
                      ivR =  mergedArgsort( 8); 
                 }
                 if (totalDigits >  8) && ( totalDigits <=  16) { 
                      ivR = mergedArgsort(16); 
                 }
                 if (totalDigits >  16) && ( totalDigits <=  32) { 
                      ivR = mergedArgsort(32); 
                 }
             } catch e: Error {
                  smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                    return "Error: %t".format(e.message());
             }

             var tmpedges = srcR[ivR]; 
             srcR=tmpedges;
             tmpedges = dstR[ivR]; 
             dstR=tmpedges;
             return "success";

          }// end combine_sortR


          proc set_neighbourR(){
             for i in 0..Ne-1 do {
                neighbourR[srcR[i]]+=1;
                if (start_iR[srcR[i]] ==-1){
                    start_iR[srcR[i]]=i;
                }
             }
          }
          proc RCM_u() throws {
            
              var cmary: [0..Nv-1] int;
              var indexary:[0..Nv-1] int;
              var depth:[0..Nv-1] int;
              depth=-1;
              proc smallVertex() :int {
                    var tmpmindegree=1000000:int;
                    var minindex=0:int;
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

              //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
              //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
              var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
              var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
              SetCurF.add(x);
              var numCurF=1:int;
              var GivenRatio=0.25:int;
              var topdown=0:int;
              var bottomup=0:int;
              var LF=1:int;
              var cur_level=0:int;
          
              while (numCurF>0) {
                    //writeln("SetCurF=");
                    //writeln(SetCurF);
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

                           proc xlocal(x :int, low:int, high:int):bool{
                                      if (low<=x && x<=high) {
                                          return true;
                                      } else {
                                          return false;
                                      }
                           }
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
                               //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                               //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                               //   if depth[i]==-1 {
                               //      UnVisitedSet.add(i);
                               //   }
                               //}
                               forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                               //forall i in UnVisitedSet  with (ref SetNextF) {
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
                               //UnVisitedSet.clear();
                               //forall i in vertexBeginR..vertexEndR with (ref UnVisitedSet) {
                               //   if depth[i]==-1 {
                               //      UnVisitedSet.add(i);
                               //   }
                               //}
                               forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                               //forall i in UnVisitedSet  with (ref SetNextF) {
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
                    //numCurF=SetNextF.getSize();
                    numCurF=SetNextF.size;

                    if (numCurF>0) {
                        //var tmpary:[0..numCurF-1] int;
                        var sortary:[0..numCurF-1] int;
                        var numary:[0..numCurF-1] int;
                        var tmpa=0:int;
                        var tmpary=SetNextF.toArray();
                        //forall (a,b)  in zip (tmpary,SetNextF.toArray()) {
                        //    a=b;
                        //}
                        forall i in 0..numCurF-1 {
                             numary[i]=neighbour[tmpary[i]];
                        }

                        var tmpiv = argsortDefault(numary);
                        sortary=tmpary[tmpiv];
                        cmary[currentindex+1..currentindex+numCurF]=sortary;
                        currentindex=currentindex+numCurF;
                    }


                    //SetCurF<=>SetNextF;
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
 
              return "success";
          }//end RCM_u


          coforall loc in Locales  {
              on loc {
                  forall i in srcR.localSubdomain(){
                        srcR[i]=dst[i];
                        dstR[i]=src[i];
                   }
              }
          }
          combine_sortR();
          set_neighbourR();

          if (DegreeSortFlag>0) {
             var DegreeArray=makeDistArray(Nv,int);
             var VertexArray=makeDistArray(Nv,int);
             var tmpedge=makeDistArray(Ne,int);
             coforall loc in Locales  {
                on loc {
                  forall i in neighbour.localSubdomain(){
                        DegreeArray[i]=neighbour[i]+neighbourR[i];
                        //writeln("Degree of vertex ",i," =",DegreeArray[i]," =",neighbour[i]," +",neighbourR[i]);
                   }
                }
             }
             //writeln("degree array=",DegreeArray);
             var tmpiv = argsortDefault(DegreeArray);
             forall i in 0..Nv-1 {
                 VertexArray[tmpiv[i]]=i;
                 //writeln("Old vertex",tmpiv[i], " -> ",i);
             }
             //writeln("relabeled vertex array=",VertexArray);
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
                           //var tmpx=src[i];
                           src[i]<=>dst[i];
                           //dst[i]=tmpx;
                        }
                   }
                }
             }


             combine_sort();
             neighbour=0;
             start_i=-1;
             set_neighbour();


             coforall loc in Locales  {
               on loc {
                  forall i in srcR.localSubdomain(){
                        srcR[i]=dst[i];
                        dstR[i]=src[i];
                   }
               }
             }
             combine_sortR();
             neighbourR=0;
             start_iR=-1;
             set_neighbourR();

          }
          if (RCMFlag>0) {
             RCM_u();
             neighbour=0;
             start_i=-1;
             combine_sort();
             set_neighbour();
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
              combine_sortR();
              set_neighbourR();

          }   
      }//end of undirected
      else {

        if (RCMFlag>0) {
           RCM();
           neighbour=0;
           start_i=-1;
           combine_sort();
           set_neighbour();

        }

      }


      var ewName ,vwName:string;
      if (weighted!=0) {
        fillInt(v_weight,1,1000);
        //fillRandom(v_weight,0,100);
        ewName = st.nextName();
        vwName = st.nextName();
        var vwEntry = new shared SymEntry(v_weight);
        var ewEntry = new shared SymEntry(e_weight);
        st.addEntry(vwName, vwEntry);
        st.addEntry(ewName, ewEntry);
      }
      var srcName = st.nextName();
      var dstName = st.nextName();
      var startName = st.nextName();
      var neiName = st.nextName();
      var srcEntry = new shared SymEntry(src);
      var dstEntry = new shared SymEntry(dst);
      var startEntry = new shared SymEntry(start_i);
      var neiEntry = new shared SymEntry(neighbour);
      st.addEntry(srcName, srcEntry);
      st.addEntry(dstName, dstEntry);
      st.addEntry(startName, startEntry);
      st.addEntry(neiName, neiEntry);
      var sNv=Nv:string;
      var sNe=Ne:string;
      var sDirected=directed:string;
      var sWeighted=weighted:string;

      var srcNameR, dstNameR, startNameR, neiNameR:string;
      if (directed!=0) {//for directed graph
          repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) ;
          if (weighted!=0) {// for weighted graph
              repMsg +=  '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } 
      } else {//for undirected graph

          srcNameR = st.nextName();
          dstNameR = st.nextName();
          startNameR = st.nextName();
          neiNameR = st.nextName();
          var srcEntryR = new shared SymEntry(srcR);
          var dstEntryR = new shared SymEntry(dstR);
          var startEntryR = new shared SymEntry(start_iR);
          var neiEntryR = new shared SymEntry(neighbourR);
          st.addEntry(srcNameR, srcEntryR);
          st.addEntry(dstNameR, dstEntryR);
          st.addEntry(startNameR, startEntryR);
          st.addEntry(neiNameR, neiEntryR);
          repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) +
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) +
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) +
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) ;
          if (weighted!=0) {// for weighted graph
              repMsg +=  '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);
          } 

      }
      timer.stop();
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$Sorting Edges takes ", timer.elapsed()," $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      outMsg="Sorting Edges takes "+ timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }




  proc segrmatgenMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;
      var (slgNv, sNe_per_v, sp, sdire,swei,RCMs )
          = payload.splitMsgToTuple(6);
      //writeln(slgNv, sNe_per_v, sp, sdire,swei,rest);
      var lgNv = slgNv: int;
      var Ne_per_v = sNe_per_v: int;
      var p = sp: real;
      var directed=sdire : int;
      var weighted=swei : int;
      var RCMFlag=RCMs : int;

      var Nv = 2**lgNv:int;
      // number of edges
      var Ne = Ne_per_v * Nv:int;

      var timer:Timer;
      timer.clear();
      timer.start();
      var n_vertices=Nv;
      var n_edges=Ne;
      var src=makeDistArray(Ne,int);
      var dst=makeDistArray(Ne,int);
      var neighbour=makeDistArray(Nv,int);
      var start_i=makeDistArray(Nv,int);
    
      var iv=makeDistArray(Ne,int);

      //var e_weight=makeDistArray(Nv,int);
      //var v_weight=makeDistArray(Nv,int);


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
      //start_i=-1;
      //neighbour=0;
      //src=1;
      //dst=1;
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
             //src=src%Nv;
             //dst=dst%Nv;

             //remove self loop
             //src=src+(src==dst);
             //src=src%Nv;
      }//end rmat_gen
      
      proc combine_sort() throws {
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;

             for (bitWidth, ary, neg) in zip(bitWidths, [src,dst], negs) {
                       (bitWidth, neg) = getBitWidth(ary); 
                       totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
             }
             proc mergedArgsort(param numDigits) throws {
                    //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
                    //             + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
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
                    var tmpiv = argsortDefault(merged);
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
                 if (totalDigits >32)  {
                       return "Error, TotalDigits >32";
                 }

             } catch e: Error {
                  smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                    return "Error: %t".format(e.message());
             }
             var tmpedges=src[iv];
             src=tmpedges;
             tmpedges=dst[iv];
             dst=tmpedges;
             // we need to change the weight order too to make them consistent 
             //if (weighted){
             //   tmpedges=e_weight[iv];
             //   e_weight=tmpedges;
             //}
             return "success";

      }//end combine_sort

      proc set_neighbour(){
             for i in 0..Ne-1 do {
                 neighbour[src[i]]+=1;
                 if (start_i[src[i]] ==-1){
                      start_i[src[i]]=i;
                      //writeln("assign index ",i, " to vertex ",src[i]);
                 }
             }
      }
      proc set_common_symtable(): string throws {
             srcName = st.nextName();
             dstName = st.nextName();
             startName = st.nextName();
             neiName = st.nextName();
             var srcEntry = new shared SymEntry(src);
             var dstEntry = new shared SymEntry(dst);
             var startEntry = new shared SymEntry(start_i);
             var neiEntry = new shared SymEntry(neighbour);
             st.addEntry(srcName, srcEntry);
             st.addEntry(dstName, dstEntry);
             st.addEntry(startName, startEntry);
             st.addEntry(neiName, neiEntry);
             sNv=Nv:string;
             sNe=Ne:string;
             sDirected=directed:string;
             sWeighted=weighted:string;
             return "success";
      }




      proc RCM() throws {
            
          var cmary: [0..Nv-1] int;
          var indexary:[0..Nv-1] int;
          var depth:[0..Nv-1] int;
          depth=-1;
          proc smallVertex() :int {
                var tmpmindegree=1000000:int;
                var minindex=0:int;
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

          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(x);
          var numCurF=1:int;
          var GivenRatio=0.021:int;
          var topdown=0:int;
          var bottomup=0:int;
          var LF=1:int;
          var cur_level=0:int;
          
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
                coforall loc in Locales  with (ref SetNextF,+ reduce topdown, + reduce bottomup) {
                   on loc {
                       ref srcf=src;
                       ref df=dst;
                       ref nf=neighbour;
                       ref sf=start_i;

                       var edgeBegin=src.localSubdomain().low;
                       var edgeEnd=src.localSubdomain().high;
                       var vertexBegin=src[edgeBegin];
                       var vertexEnd=src[edgeEnd];

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }
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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
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

                    var tmpiv = argsortDefault(numary);
                    sortary=tmpary[tmpiv];
                    cmary[currentindex+1..currentindex+numCurF]=sortary;
                    currentindex=currentindex+numCurF;
                }


                //SetCurF<=>SetNextF;
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

          return "success";
      }//end RCM


      if (directed!=0) {// for directed graph
          if (weighted!=0) { // for weighted graph
             //var e_weight: [0..Ne-1] int;
             //var v_weight: [0..Nv-1] int;
             var e_weight = makeDistArray(Ne,int);
             var v_weight = makeDistArray(Nv,int);
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             var outMsg="RMAT generate the graph takes "+timer.elapsed():string;
             smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             timer.clear();
             timer.start();
             combine_sort();
             set_neighbour();


             if (RCMFlag>0) {
                    RCM();
                    neighbour=0;
                    start_i=-1;
                    combine_sort();
                    set_neighbour();
             }


             var ewName ,vwName:string;
             fillInt(e_weight,1,1000);
             //fillRandom(e_weight,0,100);
             fillInt(v_weight,1,1000);
             //fillRandom(v_weight,0,100);
             ewName = st.nextName();
             vwName = st.nextName();
             var vwEntry = new shared SymEntry(v_weight);
             var ewEntry = new shared SymEntry(e_weight);
             try! st.addEntry(vwName, vwEntry);
             try! st.addEntry(ewName, ewEntry);
      
             set_common_symtable();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);

          } else {
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$$$$$$ RMAT generate the graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             var outMsg="RMAT generate the graph takes "+timer.elapsed():string;
             smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             timer.clear();
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();

             if (RCMFlag>0) {
                    RCM();
                    neighbour=0;
                    start_i=-1;
                    combine_sort();
                    set_neighbour();
             }

             set_common_symtable();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + '+ ' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) ; 
          }
      }// end for directed graph
      else {
          // only for undirected graph, we only declare R variables here
          var srcR=makeDistArray(Ne,int);
          var dstR=makeDistArray(Ne,int);
          var neighbourR=makeDistArray(Nv,int);
          var start_iR=makeDistArray(Nv,int);
          ref  ivR=iv;



          proc RCM_u() throws {
            
              var cmary: [0..Nv-1] int;
              var indexary:[0..Nv-1] int;
              var depth:[0..Nv-1] int;
              depth=-1;
              proc smallVertex() :int {
                    var tmpmindegree=1000000:int;
                    var minindex=0:int;
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

              //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
              //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
              var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
              var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
              SetCurF.add(x);
              var numCurF=1:int;
              var GivenRatio=0.25:int;
              var topdown=0:int;
              var bottomup=0:int;
              var LF=1:int;
              var cur_level=0:int;
          
              while (numCurF>0) {
                    //writeln("SetCurF=");
                    //writeln(SetCurF);
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

                           proc xlocal(x :int, low:int, high:int):bool{
                                      if (low<=x && x<=high) {
                                          return true;
                                      } else {
                                          return false;
                                      }
                           }
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
                               //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                               //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                               //   if depth[i]==-1 {
                               //      UnVisitedSet.add(i);
                               //   }
                               //}
                               forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                               //forall i in UnVisitedSet  with (ref SetNextF) {
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
                               //UnVisitedSet.clear();
                               //forall i in vertexBeginR..vertexEndR with (ref UnVisitedSet) {
                               //   if depth[i]==-1 {
                               //      UnVisitedSet.add(i);
                               //   }
                               //}
                               forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                               //forall i in UnVisitedSet  with (ref SetNextF) {
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
                    //numCurF=SetNextF.getSize();
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

                        var tmpiv = argsortDefault(numary);
                        sortary=tmpary[tmpiv];
                        cmary[currentindex+1..currentindex+numCurF]=sortary;
                        currentindex=currentindex+numCurF;
                    }


                    //SetCurF<=>SetNextF;
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
 
              return "success";
          }//end RCM_u



          coforall loc in Locales  {
              on loc {
                  forall i in srcR.localSubdomain(){
                        srcR[i]=dst[i];
                        dstR[i]=src[i];
                   }
              }
          }
          coforall loc in Locales  {
              on loc {
                           forall i in start_iR.localSubdomain() {
                                 start_iR[i]=-1;
                           }       
                           forall i in neighbourR.localSubdomain() {
                                 neighbourR[i]=0;
                           }       
              }
          }
          //start_iR=-1;
          //lengthR=0;
          //neighbourR=0;
          var srcNameR, dstNameR, startNameR, neiNameR:string;
        
          proc combine_sortR() throws {
             /* we cannot use the coargsort version because it will break the memory limit */
             param bitsPerDigit = RSLSD_bitsPerDigit;
             var bitWidths: [0..1] int;
             var negs: [0..1] bool;
             var totalDigits: int;
             var size=Ne: int;
             for (bitWidth, ary, neg) in zip(bitWidths, [srcR,dstR], negs) {
                 (bitWidth, neg) = getBitWidth(ary); 
                 totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;

             }
             proc mergedArgsort(param numDigits) throws {
               //overMemLimit(((4 + 3) * size * (numDigits * bitsPerDigit / 8))
               //          + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
               var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
               var curDigit = numDigits - totalDigits;
               for (ary , nBits, neg) in zip([srcR,dstR], bitWidths, negs) {
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
               var tmpiv = argsortDefault(merged);
               return tmpiv;
             } 

             try {
                 if totalDigits <=  2 { 
                      ivR = mergedArgsort( 2); 
                 }
                 if (totalDigits >  2) && ( totalDigits <=  8) { 
                      ivR =  mergedArgsort( 8); 
                 }
                 if (totalDigits >  8) && ( totalDigits <=  16) { 
                      ivR = mergedArgsort(16); 
                 }
                 if (totalDigits >  16) && ( totalDigits <=  32) { 
                      ivR = mergedArgsort(32); 
                 }
             } catch e: Error {
                  smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                    return "Error: %t".format(e.message());
             }
             var tmpedges=srcR[ivR];
             srcR=tmpedges;
             tmpedges = dstR[ivR]; 
             dstR=tmpedges;
             return "success";
             

          }// end combine_sortR

          proc    set_neighbourR(){
             for i in 0..Ne-1 do {
                neighbourR[srcR[i]]+=1;
                if (start_iR[srcR[i]] ==-1){
                    start_iR[srcR[i]]=i;
                }
             }
          }

          proc   set_common_symtableR():string throws {
          //proc   set_common_symtableR() {
             srcNameR = st.nextName();
             dstNameR = st.nextName();
             startNameR = st.nextName();
             neiNameR = st.nextName();
             var srcEntryR = new shared SymEntry(srcR);
             var dstEntryR = new shared SymEntry(dstR);
             var startEntryR = new shared SymEntry(start_iR);
             var neiEntryR = new shared SymEntry(neighbourR);
             st.addEntry(srcNameR, srcEntryR);
             st.addEntry(dstNameR, dstEntryR);
             st.addEntry(startNameR, startEntryR);
             st.addEntry(neiNameR, neiEntryR);
             return "success";
          }


          if (weighted!=0) {
             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$$$$$$ RMAT graph generating takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             var outMsg="RMAT generate the graph takes "+timer.elapsed():string;
             smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             // writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             timer.clear();
             timer.start();
             combine_sort();
             set_neighbour();


             coforall loc in Locales  {
                       on loc {
                           forall i in srcR.localSubdomain() {
                                 srcR[i]=dst[i];
                                 dstR[i]=src[i];
                           }       
                       }
             }
             //srcR = dst;
             //dstR = src;
             //twostep_sortR(); 
             combine_sortR();
             set_neighbourR();

             if (RCMFlag>0) {
                    RCM_u();
                    neighbour=0;
                    start_i=-1;
                    combine_sort();
                    set_neighbour();
                    coforall loc in Locales  {
                              on loc {
                                  forall i in srcR.localSubdomain() {
                                        srcR[i]=dst[i];
                                        dstR[i]=src[i];
                                  }       
                              }
                    }
                    neighbourR=0;
                    start_iR=-1;
                    combine_sortR();
                    set_neighbourR();

             }
             //only for weighted  graph
             var ewName ,vwName:string;
             var e_weight = makeDistArray(Ne,int);
             var v_weight = makeDistArray(Nv,int);
             //var e_weight: [0..Ne-1] int;
             //var v_weight: [0..Nv-1] int;

             fillInt(e_weight,1,1000);
             //fillRandom(e_weight,0,100);
             fillInt(v_weight,1,1000);
             //fillRandom(v_weight,0,100);
             ewName = st.nextName();
             vwName = st.nextName();
             var vwEntry = new shared SymEntry(v_weight);
             var ewEntry = new shared SymEntry(e_weight);
             st.addEntry(vwName, vwEntry);
             st.addEntry(ewName, ewEntry);
             // end of weighted!=0
      
             set_common_symtable();
             set_common_symtableR();
 
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) + 
                    '+created ' + st.attrib(vwName)    + '+created ' + st.attrib(ewName);


          } else {

             rmat_gen();
             timer.stop();
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$$$$$$ RMAT graph generating takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
             var outMsg="RMAT generate the graph takes "+timer.elapsed():string;
             smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
             timer.clear();
             timer.start();
             //twostep_sort();
             combine_sort();
             set_neighbour();


             coforall loc in Locales  {
                       on loc {
                           forall i in srcR.localSubdomain() {
                                 srcR[i]=dst[i];
                                 dstR[i]=src[i];
                           }       
                       }
             }
             combine_sortR();
             set_neighbourR();
             if (RCMFlag>0) {
                    RCM_u();
                    neighbour=0;
                    start_i=-1;
                    combine_sort();
                    set_neighbour();
                    coforall loc in Locales  {
                              on loc {
                                  forall i in srcR.localSubdomain() {
                                        srcR[i]=dst[i];
                                        dstR[i]=src[i];
                                  }       
                              }
                    }
                    neighbourR=0;
                    start_iR=-1;
                    combine_sortR();
                    set_neighbourR();

             }

             set_common_symtable();
             set_common_symtableR();
             repMsg =  sNv + '+ ' + sNe + '+ ' + sDirected + ' +' + sWeighted +
                    '+created ' + st.attrib(srcName)   + '+created ' + st.attrib(dstName) + 
                    '+created ' + st.attrib(startName) + '+created ' + st.attrib(neiName) + 
                    '+created ' + st.attrib(srcNameR)   + '+created ' + st.attrib(dstNameR) + 
                    '+created ' + st.attrib(startNameR) + '+created ' + st.attrib(neiNameR) ; 


          }// end unweighted graph
      }// end undirected graph
      timer.stop();
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$$$$$$ sorting RMAT graph takes ",timer.elapsed(), "$$$$$$$$$$$$$$$$$$");
      var outMsg="sorting RMAT graph takes "+timer.elapsed():string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);      
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      //writeln("$$$$$$$$$$$$  $$$$$$$$$$$$$$$$$$$$$$$");
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }



  proc segBFSMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var repMsg: string;
      //var (n_verticesN,n_edgesN,directedN,weightedN,srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN )
      //    = payload.decode().splitMsgToTuple(10);
      var (RCMs,n_verticesN,n_edgesN,directedN,weightedN,restpart )
          = payload.splitMsgToTuple(6);
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
      //var depth=-1: [0..Nv-1] int;
      var root:int;
      var srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN :string;
      var srcRN, dstRN, startRN, neighbourRN:string;
       


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
          //writeln("CMaxSize=",CMaxSize);
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          //var localArrayNG=makeDistArray(numLocales*CMaxSize,int);// next frontier in the same locale
          //var sendArrayG=makeDistArray(numLocales*CMaxSize,int);// next frontier in other locales
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          //var LPNG=makeDistArray(numLocales,int);// next frontier pointer to current position
          //var SPG=makeDistArray(numLocales,int);// receive buffer
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          //LPNG=0;
          //SPG=0;
          RPG=0;

          proc xlocal(x :int, low:int, high:int):bool{
                     if (low<=x && x<=high) {
                            return true;
                     } else {
                            return false;
                     }
          }

          proc xremote(x :int, low:int, high:int):bool{
                     if (low>=x || x>=high) {
                            return true;
                     } else {
                            return false;
                     }
          }
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
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                                    //writeln("2 My locale=", here.id," Add ", j, " into local");
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
                                                    RemoteSet.add(j);
                                                    //writeln("3 My locale=", here.id," Add ", j, " into remote");
                                               }
                                         }
                                  }
                              } 
                       
                              if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  //writeln("6-0 My locale=", here.id," there are remote element =",RemoteSet);
                                  coforall localeNum in 0..numLocales-1  { 
                                         var ind=0:int;
                                         //for k in RemoteSet with ( +reduce ind) (var agg= newDstAggregator(int)) {
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                //writeln("6-2 My locale=", here.id," test remote element ", k);
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                     //writeln("6 My locale=", here.id,"send", k, "to locale= ",localeNum," number=", ind, " send element=", recvArrayG[localeNum*numLocales*CMaxSize+ here.id*CMaxSize+ind-1]);
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
                              //writeln("7-0 My locale=", here.id,"  a=",a, " b=",b);
                              a=b;
                       }
                       var tmp=0;
                       for i in LocalSet {
                              //writeln("7-1 My locale=", here.id,"  element i=",i," tmp=",tmp);
                              //localArrayG[mystart+tmp]=i;
                              //writeln("7-2 My locale=", here.id,"  local array [tmp]=",localArrayG[mystart+tmp]," tmp=",tmp);
                              tmp+=1;
                       }
                       //writeln("7 My locale=", here.id,"  local set=",LocalSet, "to local array and size= ",LocalSet.size, " local array=",localArrayG[mystart..mystart+LocalSet.size-1]);
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                   //LPNG[here.id]=0;
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
                           //writeln("8 My locale=", here.id," after colloect array=",localArrayG[mystart..mystart+LPG[here.id]-1]);
                       }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              //writeln("10-0 LPG=",LPG);
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       //writeln("10  locale ",iL, " has ",LPG[iL], " elements");
                       numCurF=1;
                       break;
                   }
              }
              RPG=0;
              cur_level+=1;
              //writeln("cur level=",cur_level);
          }//end while  
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          writeln("Local Ratio=", (TotalLocal):real/(TotalLocal+TotalRemote):real,"Total Local Access=",TotalLocal," , Total Remote Access=",TotalRemote);
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
          //writeln("CMaxSize=",CMaxSize);
          var localArrayG=makeDistArray(numLocales*CMaxSize,int);//current frontier elements
          //var localArrayNG=makeDistArray(numLocales*CMaxSize,int);// next frontier in the same locale
          //var sendArrayG=makeDistArray(numLocales*CMaxSize,int);// next frontier in other locales
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          //var LPNG=makeDistArray(numLocales,int);// next frontier pointer to current position
          //var SPG=makeDistArray(numLocales,int);// receive buffer
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          //LPNG=0;
          //SPG=0;
          RPG=0;

          proc xlocal(x :int, low:int, high:int):bool{
                     if (low<=x && x<=high) {
                            return true;
                     } else {
                            return false;
                     }
          }

          proc xremote(x :int, low:int, high:int):bool{
                     if (low>=x || x>=high) {
                            return true;
                     } else {
                            return false;
                     }
          }
          coforall loc in Locales   {
              on loc {
                 if (xlocal(root,vertexBeginG[here.id],vertexEndG[here.id]) || 
                                 xlocal(root,vertexBeginRG[here.id],vertexEndRG[here.id])) {
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
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                                    //writeln("2 My locale=", here.id," Add ", j, " into local");
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
                                                    RemoteSet.add(j);
                                                    //writeln("3 My locale=", here.id," Add ", j, " into remote");
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
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                                    //writeln("4 reverse My locale=", here.id,"Add ", j, "into local");
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
                                                    RemoteSet.add(j);
                                                    //writeln("5 reverse My locale=", here.id,"Add ", j, "into remote");
                                               }
                                         }
                                  }
                              }
                       
                              if (RemoteSet.size>0) {//there is vertex to be sent
                                  remoteNum[here.id]+=RemoteSet.size;
                                  //writeln("6-0 My locale=", here.id," there are remote element =",RemoteSet);
                                  coforall localeNum in 0..numLocales-1  { 
                                         var ind=0:int;
                                         //for k in RemoteSet with ( +reduce ind) (var agg= newDstAggregator(int)) {
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                //writeln("6-2 My locale=", here.id," test remote element ", k);
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     ind+=1;
                                                     
                                                     //writeln("6 My locale=", here.id,"send", k, "to locale= ",localeNum," number=", ind, " send element=", recvArrayG[localeNum*numLocales*CMaxSize+ here.id*CMaxSize+ind-1]);
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
                              //writeln("7-0 My locale=", here.id,"  a=",a, " b=",b);
                              a=b;
                       }
                       var tmp=0;
                       for i in LocalSet {
                              //writeln("7-1 My locale=", here.id,"  element i=",i," tmp=",tmp);
                              //localArrayG[mystart+tmp]=i;
                              //writeln("7-2 My locale=", here.id,"  local array [tmp]=",localArrayG[mystart+tmp]," tmp=",tmp);
                              tmp+=1;
                       }
                       //writeln("7 My locale=", here.id,"  local set=",LocalSet, "to local array and size= ",LocalSet.size, " local array=",localArrayG[mystart..mystart+LocalSet.size-1]);
                   }
                   LocalSet.clear();
                   RemoteSet.clear();
                   //LPNG[here.id]=0;
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
                           //writeln("8 My locale=", here.id," after colloect array=",localArrayG[mystart..mystart+LPG[here.id]-1]);
                       }
                         
                   }
                  }//end on loc
              }//end coforall loc
              numCurF=0;
              //writeln("10-0 LPG=",LPG);
              for iL in 0..(numLocales-1)  {
                   if LPG[iL] >0 {
                       //writeln("10  locale ",iL, " has ",LPG[iL], " elements");
                       numCurF=1;
                       break;
                   }
              }
              RPG=0;
              cur_level+=1;
              //writeln("cur level=",cur_level);
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          writeln("Local Ratio=", (TotalLocal):real/(TotalLocal+TotalRemote):real,"Total Local Access=",TotalLocal," , Total Remote Access=",TotalRemote);
          return "success";
      }//end of _d1_bfs_kernel_u

      proc fo_bag_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          //var SetCurF: domain(int);//use domain to keep the current frontier
          //var SetNextF:domain(int);//use domain to keep the next frontier
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          //var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          //var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;
          //var GivenRatio=0.22:int;
          //writeln("THE GIVEN RATIO IS ");
          //writeln(GivenRatio);

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }
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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                //SetCurF=SetNextF;
                //SetCurF.clear();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$number of top down = ",topdown, " number of bottom up=", bottomup,"$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of fo_bag_bfs_kernel_u


      proc fo_bag_bfs_kernel(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          //var SetCurF: domain(int);//use domain to keep the current frontier
          //var SetNextF:domain(int);//use domain to keep the next frontier
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          //var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          //var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }
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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                //SetCurF=SetNextF;
                //SetCurF.clear();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$number of top down = ",topdown, " number of bottom up=", bottomup,"$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of fo_bag_bfs_kernel


      proc fo_set_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          //var SetCurF: domain(int);//use domain to keep the current frontier
          //var SetNextF:domain(int);//use domain to keep the next frontier
          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }

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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
                numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                SetCurF=SetNextF;
                //SetCurF.clear();
                //SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$number of top down = ",topdown, " number of bottom up=", bottomup,"$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of fo_set_bfs_kernel_u




      proc fo_domain_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF: domain(int);//use domain to keep the current frontier
          var SetNextF:domain(int);//use domain to keep the next frontier
          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          //var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          //var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var topdown=0:int;
          var bottomup=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }

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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           forall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //forall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
                numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                SetCurF=SetNextF;
                //SetCurF.clear();
                //SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$number of top down = ",topdown, " number of bottom up=", bottomup,"$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of fo_domain_bfs_kernel_u



      proc fo_d1_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
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
          //var localArrayNG=makeDistArray(numLocales*CMaxSize,int);// next frontier in the same locale
          //var sendArrayG=makeDistArray(numLocales*CMaxSize,int);// next frontier in other locales
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          //var LPNG=makeDistArray(numLocales,int);// next frontier pointer to current position
          //var SPG=makeDistArray(numLocales,int);// receive buffer
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          //LPNG=0;
          //SPG=0;
          RPG=0;

          proc xlocal(x :int, low:int, high:int):bool{
                     if (low<=x && x<=high) {
                            return true;
                     } else {
                            return false;
                     }
          }

          proc xremote(x :int, low:int, high:int):bool{
                     if (low>=x || x>=high) {
                            return true;
                     } else {
                            return false;
                     }
          }
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


                   //var aggdst= newDstAggregator(int);
                   //var aggsrc= newSrcAggregator(int);
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
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
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
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
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
                                         //for k in RemoteSet with ( +reduce ind) (var agg= newDstAggregator(int)) {
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     //recvArrayG[localeNum*numLocales*CMaxSize+
                                                     //                    here.id*CMaxSize+ind]=k;
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
                       //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                       //forall i in BoundBeginG[here.id]..BoundEndG[here.id] with (ref UnVisitedSet) {
                       //       if depth[i]==-1 {
                       //          UnVisitedSet.add(i);
                       //       }
                       //}
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
                       //forall i in UnVisitedSet  with (ref LocalSet, ref RemoteSet)  {
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
                   //LPNG[here.id]=0;
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
                       //numCurF=1;
                       numCurF+=LPG[iL];
                       //break;
                   }
              }
              RPG=0;
              cur_level+=1;
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$number of top-down = ", topdown, " number of bottom-up=",bottomup, "$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          writeln("Local Ratio=", (TotalLocal):real/(TotalLocal+TotalRemote):real,"Total Local Access=",TotalLocal," , Total Remote Access=",TotalRemote);
          return "success";
      }//end of fo_d1_bfs_kernel_u


      proc co_bag_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          //var SetCurF: domain(int);//use domain to keep the current frontier
          //var SetNextF:domain(int);//use domain to keep the next frontier
          var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          //var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          //var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var bottomup=0:int;
          var topdown=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }
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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}

                           //coforall i in UnVisitedSet with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //coforall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                //SetCurF=SetNextF;
                //SetCurF.clear();
                SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$number of top-down = ", topdown, " number of bottom-up=",bottomup, "$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of co_bag_bfs_kernel_u




      proc co_set_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          //var SetCurF: domain(int);//use domain to keep the current frontier
          //var SetNextF:domain(int);//use domain to keep the next frontier
          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;

          var bottomup=0:int;
          var topdown=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }

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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           coforall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //coforall i in UnVisitedSet  with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //coforall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
                numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                SetCurF=SetNextF;
                //SetCurF.clear();
                //SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$number of top-down = ", topdown, " number of bottom-up=",bottomup, "$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          return "success";
      }//end of co_set_bfs_kernel_u




      proc co_domain_bfs_kernel_u(nei:[?D1] int, start_i:[?D2] int,src:[?D3] int, dst:[?D4] int,
                        neiR:[?D11] int, start_iR:[?D12] int,srcR:[?D13] int, dstR:[?D14] int, 
                        LF:int,GivenRatio:real):string throws{
          var cur_level=0;
          var SetCurF: domain(int);//use domain to keep the current frontier
          var SetNextF:domain(int);//use domain to keep the next frontier
          //var SetCurF=  new DistBag(int,Locales);//use bag to keep the current frontier
          //var SetNextF=  new DistBag(int,Locales); //use bag to keep the next frontier
          //var SetCurF= new set(int,parSafe = true);//use set to keep the current frontier
          //var SetNextF=new set(int,parSafe = true);//use set to keep the next fromtier
          SetCurF.add(root);
          var numCurF=1:int;
          var bottomup=0:int;
          var topdown=0:int;

          //while (!SetCurF.isEmpty()) {
          while (numCurF>0) {
                //writeln("SetCurF=");
                //writeln(SetCurF);
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

                       proc xlocal(x :int, low:int, high:int):bool{
                                  if (low<=x && x<=high) {
                                      return true;
                                  } else {
                                      return false;
                                  }
                       }

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
                           //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                           //forall i in vertexBegin..vertexEnd  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           coforall i in vertexBegin..vertexEnd  with (ref SetNextF) {
                           //coforall i in UnVisitedSet with (ref SetNextF) {
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
                           //UnVisitedSet.clear();
                           //forall i in vertexBeginR..vertexEndR  with (ref UnVisitedSet) {
                           //   if depth[i]==-1 {
                           //      UnVisitedSet.add(i);
                           //   }
                           //}
                           coforall i in vertexBeginR..vertexEndR  with (ref SetNextF) {
                           //coforall i in UnVisitedSet  with (ref SetNextF) {
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
                //numCurF=SetNextF.getSize();
                numCurF=SetNextF.size;
                //writeln("SetCurF= ", SetCurF, " SetNextF=", SetNextF, " level ", cur_level+1," numCurf=", numCurF);
                //numCurF=SetNextF.size;
                SetCurF=SetNextF;
                //SetCurF.clear();
                //SetCurF<=>SetNextF;
                SetNextF.clear();
          }//end while  
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$number of top-down = ", topdown, " number of bottom-up=",bottomup, "$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
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
          //var localArrayNG=makeDistArray(numLocales*CMaxSize,int);// next frontier in the same locale
          //var sendArrayG=makeDistArray(numLocales*CMaxSize,int);// next frontier in other locales
          var recvArrayG=makeDistArray(numLocales*numLocales*CMaxSize,int);//hold the current frontier elements
          var LPG=makeDistArray(numLocales,int);// frontier pointer to current position
          //var LPNG=makeDistArray(numLocales,int);// next frontier pointer to current position
          //var SPG=makeDistArray(numLocales,int);// receive buffer
          var RPG=makeDistArray(numLocales*numLocales,int);//pointer to the current position can receive element
          LPG=0;
          //LPNG=0;
          //SPG=0;
          RPG=0;

          proc xlocal(x :int, low:int, high:int):bool{
                     if (low<=x && x<=high) {
                            return true;
                     } else {
                            return false;
                     }
          }

          proc xremote(x :int, low:int, high:int):bool{
                     if (low>=x || x>=high) {
                            return true;
                     } else {
                            return false;
                     }
          }
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


                   //var aggdst= newDstAggregator(int);
                   //var aggsrc= newSrcAggregator(int);
                   var LocalSet= new set(int,parSafe = true);//use set to keep the next local frontier, 
                                                             //vertex in src or srcR
                   var RemoteSet=new set(int,parSafe = true);//use set to keep the next remote frontier

                   var mystart=here.id*CMaxSize;//start index 



                   var   switchratio=(numCurF:real)/nf.size:real;
                   if (switchratio<GivenRatio) {//top down
                       topdown+=1;
                       coforall i in localArrayG[mystart..mystart+LPG[here.id]-1] 
                                                   with (ref LocalSet, ref RemoteSet)  {
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
                                               if (xlocal(j,vertexBeginG[here.id],vertexEndG[here.id]) ||
                                                   xlocal(j,vertexBeginRG[here.id],vertexEndRG[here.id])) {
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
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
                                                    //localArrayNG[mystart+LPNG[here.id]]=j; 
                                                    //LPNG[here.id]+=1;
                                                    LocalSet.add(j);
                                               } 
                                               if (xremote(j,HvertexBeginG[here.id],TvertexEndG[here.id]) ||
                                                   xremote(j,HvertexBeginRG[here.id],TvertexEndRG[here.id])) {
                                                    //sendArrayG[mystart+SPG[here.id]]=j;                 
                                                    //SPG[here.id]+=1;
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
                                         //for k in RemoteSet with ( +reduce ind) (var agg= newDstAggregator(int)) {
                                         var agg= newDstAggregator(int); 
                                         for k in RemoteSet {
                                                if (xlocal(k,vertexBeginG[localeNum],vertexEndG[localeNum])||
                                                    xlocal(k,vertexBeginRG[localeNum],vertexEndRG[localeNum])){
                                                     agg.copy(recvArrayG[localeNum*numLocales*CMaxSize+
                                                                         here.id*CMaxSize+ind] ,k);
                                                     //recvArrayG[localeNum*numLocales*CMaxSize+
                                                     //                    here.id*CMaxSize+ind]=k;
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
                                    //coforall j in localArrayG[mystart..mystart+LPG[i]-1] with (ref returnval){
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

                       //var UnVisitedSet= new set(int,parSafe = true);//use set to keep the unvisited vertices
                       //forall i in BoundBeginG[here.id]..BoundEndG[here.id]  with (ref UnVisitedSet) {
                       //       if depth[i]==-1 {
                       //          UnVisitedSet.add(i);
                       //       }
                       //}
                       coforall i in BoundBeginG[here.id]..BoundEndG[here.id]
                                                   with (ref LocalSet, ref RemoteSet)  {
                       //coforall i in UnVisitedSet with (ref LocalSet, ref RemoteSet)  {
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
                   //LPNG[here.id]=0;
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
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$Search Radius = ", cur_level+1,"$$$$$$$$$$$$$$$$$$$$$$");
          var outMsg="Search Radius = "+ (cur_level+1):string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          outMsg= "number of top-down = "+ topdown:string+ " number of bottom-up="+bottomup:string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          //writeln("$$$$$$$$number of top-down = ", topdown, " number of bottom-up=",bottomup, "$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          //writeln("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
          var TotalLocal=0:int;
          var TotalRemote=0:int;
          for i in 0..numLocales-1 {
            TotalLocal+=localNum[i];
            TotalRemote+=remoteNum[i];
          }
          outMsg="Local Ratio="+ ((TotalLocal):real/(TotalLocal+TotalRemote):real):string + "Total Local Access=" + TotalLocal:string +" Total Remote Access=" + TotalRemote:string;
          smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
          //writeln("Local Ratio=", (TotalLocal):real/(TotalLocal+TotalRemote):real,"Total Local Access=",TotalLocal," , Total Remote Access=",TotalRemote);
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
              //var pn = Reflection.getRoutineName();
               (srcN, dstN, startN, neighbourN,vweightN,eweightN, rootN,ratios)=
                   restpart.splitMsgToTuple(8);
              root=rootN:int;
              var GivenRatio=ratios:real;
              if (RCMFlag>0) {
                  root=0;
              }
              depth[root]=0;
              var ag = new owned SegGraphDW(Nv,Ne,Directed,Weighted,srcN,dstN,
                                 startN,neighbourN,vweightN,eweightN, st);
              fo_bag_bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,1,GivenRatio);
              repMsg=return_depth();

          } else {
              var ratios:string;

              (srcN, dstN, startN, neighbourN,rootN,ratios )=restpart.splitMsgToTuple(6);
              var ag = new owned SegGraphD(Nv,Ne,Directed,Weighted,srcN,dstN,
                      startN,neighbourN,st);
              root=rootN:int;
              var GivenRatio=ratios:real;
              if (RCMFlag>0) {
                  root=0;
              }
              depth[root]=0;
              fo_bag_bfs_kernel(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,1,GivenRatio);
              repMsg=return_depth();
          }
      }
      else {
          if (Weighted!=0) {
              var ratios:string;
              (srcN, dstN, startN, neighbourN,srcRN, dstRN, startRN, neighbourRN,vweightN,eweightN, rootN, ratios)=
                   restpart.splitMsgToTuple(12);
              var ag = new owned SegGraphUDW(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      vweightN,eweightN, st);
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
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,1,GivenRatio);
                  repMsg=return_depth();
 
              } else {// do batch test
                  depth=-1;
                  depth[root]=0;
                  timer.stop();
                  timer.clear();
                  timer.start();
                  co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                  timer.stop();
                  //writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co D Hybrid version $$$$$$$$$$$$$$$$$$");
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
              var ag = new owned SegGraphUD(Nv,Ne,Directed,Weighted,
                      srcN,dstN, startN,neighbourN,
                      srcRN,dstRN, startRN,neighbourRN,
                      st);

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
                  fo_bag_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,1,GivenRatio);
                  repMsg=return_depth();
 
              } else {// do batch test
                  timer.stop();
                  timer.clear();
                  timer.start();
                  co_d1_bfs_kernel_u(ag.neighbour.a, ag.start_i.a,ag.src.a,ag.dst.a,
                           ag.neighbourR.a, ag.start_iR.a,ag.srcR.a,ag.dstR.a,GivenRatio);
                  timer.stop();
                  var outMsg= "graph BFS takes "+timer.elapsed():string+ " for Co D Hybrid version";
                  smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),outMsg);
                  //writeln("$$$$$$$$$$$$$$$$$ graph BFS takes ",timer.elapsed(), " for Co D Hybrid version $$$$$$$$$$$$$$$$$$");
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










}


