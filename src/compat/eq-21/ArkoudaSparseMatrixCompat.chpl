module ArkoudaSparseMatrixCompat {
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
}
