
module SymArrayDmapCompat
{
    use ChplConfig;

    /*
     Available domain maps.
     */
    enum Dmap {defaultRectangular, blockDist};

    private param defaultDmap = if CHPL_COMM == "none" then Dmap.defaultRectangular
                                                       else Dmap.blockDist;
    /*
    How domains/arrays are distributed. Defaults to :enum:`Dmap.defaultRectangular` if
    :param:`CHPL_COMM=none`, otherwise defaults to :enum:`Dmap.blockDist`.
    */
    config param MyDmap:Dmap = defaultDmap;

    public use BlockDist;

    /* 
    Makes a domain distributed according to :param:`MyDmap`.

    :arg shape: size of domain in each dimension
    :type shape: int
    */
    proc makeDistDom(shape: int ...?N) {
        var rngs: N*range;
        for i in 0..#N do rngs[i] = 0..#shape[i];
        const dom = {(...rngs)};

        select MyDmap
        {
            when Dmap.defaultRectangular {
                return dom;
            }
            when Dmap.blockDist {
                if dom.size > 0 {
                    return blockDist.createDomain(dom);
                }
                // fix the annoyance about boundingBox being empty
                else {
                  return dom dmapped blockDist(boundingBox=dom.expand(1));
                }
            }
            otherwise {
                halt("Unsupported distribution " + MyDmap:string);
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
      where MyDmap != Dmap.defaultRectangular && a.isDefaultRectangular() {
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

}
