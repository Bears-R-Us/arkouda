module ArkoudaSparseMatrixCompat {
    use LayoutCS;
    import SparseMatrix.Layout;

    proc getSparseDom(param layout: Layout) {
        return new dmap(new CS(compressRows=(matLayout==Layout.CSR)));
    }

    proc getSparseDomType(param layout: Layout) type {
        return CS(compressRows=(layout==Layout.CSR));;
    }
}
