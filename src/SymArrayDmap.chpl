
module SymArrayDmap
{
    /*
     Available domain maps. Cyclic isn't regularly tested and may not work.
     */
    enum Dmap {blockDist, cyclicDist};

    config param MyDmap:Dmap = Dmap.blockDist;

    public use CyclicDist;
    public use BlockDist;

    /* 
    Makes a domain distributed according to :param:`MyDmap`.

    :arg size: size of domain
    :type size: int
    */
    proc makeDistDom(size:int) {
        select MyDmap
        {
            when Dmap.blockDist {
                if size > 0 {
                    return {0..#size} dmapped Block(boundingBox={0..#size});
                }
                // fix the annoyance about boundingBox being enpty
                else {return {0..#0} dmapped Block(boundingBox={0..0});}
            }
            when Dmap.cyclicDist {
                return {0..#size} dmapped Cyclic(startIdx=0);
            }
            otherwise {
                halt("Unsupported distribution " + MyDmap:string);
            }
        }
    }
    
    /* 
    Makes an array of specified type over a distributed domain

    :arg size: size of the domain
    :type size: int 

    :arg etype: desired type of array
    :type etype: type

    :returns: [] ?etype
    */
    proc makeDistArray(size:int, type etype) {
        var a: [makeDistDom(size)] etype;
        return a;
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
