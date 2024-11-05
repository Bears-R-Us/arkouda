module ArkoudaSparseMatrixCompat {
  use SparseBlockDist;
  use LayoutCS;
  import SparseMatrix.SpsMatUtil.Layout;
  use BlockDist;

  proc getSparseDom(param layout: Layout) {
    return new dmap(new CS(compressRows=(layout==Layout.CSR)));
  }

  proc getDenseDom(dom, localeGrid, param layout: Layout) {
    type layoutType = CS(compressRows=(layout==Layout.CSR));
    return dom dmapped new blockDist(boundingBox=dom,
                                     targetLocales=localeGrid,
                                     sparseLayoutType=layoutType);
  }

  proc SparseBlockDom.setLocalSubdomain(locIndices, loc: locale = here) {
    if loc != here then
      halt("setLocalSubdomain() doesn't currently support remote updates");
    ref myBlock = this.myLocDom!.mySparseBlock;
    if myBlock.type != locIndices.type then
      compilerError("setLocalSubdomain() expects its argument to be of type ",
                     myBlock.type:string);
    else
      myBlock = locIndices;
  }

  proc SparseBlockArr.getLocalSubarray(localeRow, localeCol) const ref {
    return this.locArr[localeRow, localeCol]!.myElems;
  }

  proc SparseBlockArr.getLocalSubarray(localeIdx) const ref {
    return this.locArr[localeIdx]!.myElems;
  }

  proc SparseBlockArr.setLocalSubarray(locNonzeroes, loc: locale = here) {
    if loc != here then
      halt("setLocalSubarray() doesn't currently support remote updates");
    ref myBlock = this.myLocArr!.myElems;
    if myBlock.type != locNonzeroes.type then
      compilerError("setLocalSubarray() expects its argument to be of type ",
                    myBlock.type:string);
    else
      myBlock.data = locNonzeroes.data;
  }

  proc SparseBlockDom.dsiTargetLocales() const ref {
    return dist.targetLocales;
  }

  proc SparseBlockArr.dsiTargetLocales() const ref {
    return dom.dsiTargetLocales();
  }

  use LayoutCS;

  proc CSDom.rows() {
    return this.rowRange;
  }

  proc CSDom.cols() {
    return this.colRange;
  }

  @chpldoc.nodoc
  iter CSDom.uidsInRowCol(rc) {
    for uid in startIdx[rc]..<startIdx[rc+1] do
      yield uid;
  }

  proc CSArr.rows() {
    return this.dom.rows();
  }

  proc CSArr.cols() {
    return this.dom.cols();
  }

  @chpldoc.nodoc
  iter CSArr.indsAndVals(rc) {
    ref dom = this.dom;
    for uid in dom.uidsInRowCol(rc) do
      yield (dom.idx[uid], this.data[uid]);
  }

  iter CSArr.colsAndVals(r) {
    if this.dom.compressRows == false then
      compilerError("Can't (efficiently) iterate over rows using a CSC layout");
    for colVal in indsAndVals(r) do
      yield colVal;
  }

  iter CSArr.rowsAndVals(c) {
    if this.dom.compressRows == true then
      compilerError("Can't (efficiently) iterate over columns using a CSR layout");
    for rowVal in indsAndVals(c) do
      yield rowVal;
  }
}
