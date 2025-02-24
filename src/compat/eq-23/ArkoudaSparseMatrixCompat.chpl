module ArkoudaSparseMatrixCompat {
    use CompressedSparseLayout;
    import SparseMatrix.SpsMatUtil.Layout;
    use BlockDist;

    proc getSparseDom(param layout: Layout) {
        select layout {
            when Layout.CSR do return new csrLayout();
            when Layout.CSC do return new cscLayout();
        }
    }

   // see: https://github.com/chapel-lang/chapel/issues/26209
    proc getDenseDom(dom, localeGrid, param layout: Layout) {
        if layout == Layout.CSR {
            return dom dmapped new blockDist(boundingBox=dom,
                                             targetLocales=localeGrid,
                                             sparseLayoutType=csrLayout);
        } else {
            return dom dmapped new blockDist(boundingBox=dom,
                                             targetLocales=localeGrid,
                                             sparseLayoutType=cscLayout);
        }
    }
}
