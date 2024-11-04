module ArkoudaSparseMatrixCompat {
    use CompressedSparseLayout;
    import SparseMatrix.SpsMatUtil.Layout;

    proc getSparseDom(param layout: Layout) {
        select layout {
            when Layout.CSR do return new csrLayout();
            when Layout.CSC do return new cscLayout();
        }
    }

    proc getSparseDomType(param layout: Layout) type {
        if layout == Layout.CSR then return csrLayout; else return cscLayout;
    }
}
