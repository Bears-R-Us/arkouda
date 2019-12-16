
module SymArrayDmap
{
    /*
    Configure MyDmap on compile line by "-s MyDmap=0" or "-s MyDmap=1"
    0 = Cyclic, 1 = Block. Cyclic may not work; we haven't tested it in a while.
    BlockDist is the default.
    */
    config param MyDmap = 1;

    public use CyclicDist;
    public use BlockDist;

    /* 
    Uses the MyDmap config param in ServerConfig.chpl::
        *if MyDmap == 0 {return (type CyclicDom(1,int(64),false));}* 

        *if MyDmap == 1 {return (type BlockDom(1,int(64),false,unmanaged DefaultDist));}*

    :arg size: size of domain
    :type size: int

    **Note**: if MyDmap does not evaluate to 0 or 1, Cyclic Distribution will be selected. 
    Cyclic Distribution is currently not fully supported.
    **Note 2**: MyDmap is by default set to 1 in ServerConfig.chpl
    */
    proc makeDistDom(size:int) {
        select MyDmap
        {
            when 0 { // Cyclic distribution
                return {0..#size} dmapped Cyclic(startIdx=0);
            }
            when 1 { // Block Distribution
                if size > 0 {
                    return {0..#size} dmapped Block(boundingBox={0..#size});
                }
                // fix the annoyance about boundingBox being enpty
                else {return {0..#0} dmapped Block(boundingBox={0..0});}
            }
            otherwise { // default to cyclic distribution
                return {0..#size} dmapped Cyclic(startIdx=0);
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