module ArkoudaSparseMatrixCompat {
    use LayoutCS;
    import SparseMatrix.SpsMatUtil.Layout;

    proc getSparseDom(param layout: Layout) {
        return new dmap(new CS(compressRows=(layout==Layout.CSR)));
    }

    proc getSparseDomType(param layout: Layout) type {
        return CS(compressRows=(layout==Layout.CSR));
    }
}
