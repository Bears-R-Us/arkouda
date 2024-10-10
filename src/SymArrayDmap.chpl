module SymArrayDmap {
    import ChplConfig;
    import SparseMatrix.layout;
    public use BlockDist;
    use LayoutCS;

    /*
        Available domain maps.
    */
    enum Dmap {defaultRectangular, blockDist};

    private param defaultDmap = if ChplConfig.CHPL_COMM == "none" then Dmap.defaultRectangular
                                                                    else Dmap.blockDist;
    config param MyDmap:Dmap = defaultDmap;

    /*
        Makes a domain distributed according to :param:`MyDmap`.

        :arg shape: size of domain in each dimension
        :type shape: int
    */
    proc makeDistDom(shape: int ...?N) {
      var rngs: N*range;
      for i in 0..#N do rngs[i] = 0..#shape[i];
      const dom = {(...rngs)};

      return makeDistDom(dom);
    }

    proc makeDistDom(dom: domain(?)) {
      select MyDmap {
        when Dmap.defaultRectangular {
          return dom;
        }
        when Dmap.blockDist {
          if dom.size > 0 {
              return blockDist.createDomain(dom);
          }
          // fix the annoyance about boundingBox being empty
          else {
            return dom dmapped new blockDist(boundingBox=dom.expand(1));
          }
        }
      }
    }

    /*
        Makes an array of specified type over a distributed domain

        :arg shape: size of the domain in each dimension
        :type shape: int

        :arg etype: desired type of array
        :type etype: type

        :returns: [] ?etype
    */
    proc makeDistArray(shape: int ...?N, type etype) throws
      where N == 1
    {
      var dom = makeDistDom((...shape));
      return dom.tryCreateArray(etype);
    }

    proc makeDistArray(shape: int ...?N, type etype) throws
      where N > 1
    {
      var a: [makeDistDom((...shape))] etype;
      return a;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular()
    {
        var res = makeDistArray((...D.shape), etype);
        res = a;
        return res;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where D.rank == 1 && (MyDmap == Dmap.defaultRectangular || !a.isDefaultRectangular())
    {
      var res = D.tryCreateArray(etype);
      res = a;
      return res;
    }

    proc makeDistArray(in a: [?D] ?etype) throws
      where D.rank > 1 && (MyDmap == Dmap.defaultRectangular || !a.isDefaultRectangular())
    {
      return a;
    }

    proc makeDistArray(D: domain(?), type etype) throws
      where D.rank == 1
    {
      var res = D.tryCreateArray(etype);
      return res;
    }

    proc makeDistArray(D: domain(?), type etype) throws
      where D.rank > 1
    {
      var res: [D] etype;
      return res;
    }

    proc makeDistArray(D: domain(?), initExpr: ?t) throws
      where D.rank == 1
    {
      return D.tryCreateArray(t, initExpr);
    }

    proc makeDistArray(D: domain(?), initExpr: ?t) throws
      where D.rank > 1
    {
      var res: [D] t = initExpr;
      return res;
    }

    /*
        Returns the type of the distributed domain

        :arg size: size of domain
        :type size: int

        :returns: type
    */
    proc makeDistDomType(size: int) type {
        return makeDistDom(size).type;
    }

    proc makeSparseDomain(size: int, param matLayout) {
      const dom = {1..size, 1..size}; // TODO: only supporting square matrices for now?
                                      // TODO: change domain to be zero based?
      select MyDmap {
        when Dmap.defaultRectangular {
          var spsDom: sparse subdomain(dom) dmapped new dmap(new CS(compressRows=(matLayout==layout.CSR)));
          return (spsDom, dom);
        }
        when Dmap.blockDist {
          const locsPerDim = sqrt(numLocales:real): int,
                grid = {0..<locsPerDim, 0..<locsPerDim},
                localeGrid = reshape(Locales[0..<grid.size], grid);

          type layoutType = CS(compressRows=(matLayout==layout.CSR));
          const DenseBlkDom = dom dmapped new blockDist(boundingBox=dom,
                                                        targetLocales=localeGrid,
                                                        sparseLayoutType=layoutType);

          var SD: sparse subdomain(DenseBlkDom);
          return (SD, DenseBlkDom);
        }
      }
    }

    proc makeSparseArray(size: int, type eltType, param matLayout) {
      const (sd, _) = makeSparseDomain(size, matLayout);
      var arr: [sd] eltType;
      return arr;
    }

}
