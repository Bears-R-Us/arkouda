BOOLEAN_FUNCS = [
    ["any", "bool", []],
    ["all", "bool", []],
    ["isSorted", "bool", []],
    ["isSortedLocally", "bool", []],
]

INDEX_FUNCS = [
    ["argmax", "d.idxType", ["int", "uint", "real", "bool"]],
    ["argmin", "d.idxType", ["int", "uint", "real", "bool"]],
]


def generate_header():
    return """module ReductionMsgFunctions
{
    use BigInteger;
    use List;
    use AryUtil;
    use ReductionMsg;
    use SliceReductionOps;
    """


def generate_footer():
    return "\n}\n"


def generate_reduction_functions():
    ret = ""

    for func, ret_type, allowed_types in BOOLEAN_FUNCS:

        where_statement = ""

        if len(allowed_types) > 0:
            where_statement += "where "
            where_statement += "||".join([f"(t=={tp})" for tp in allowed_types])

        ret += f"""
    @arkouda.registerCommand
    proc {func}All(const ref x:[?d] ?t): {ret_type} throws
    {where_statement}{{
      use SliceReductionOps;
      return {func}Slice(x, x.domain);
    }}

    @arkouda.registerCommand
    proc {func}(const ref x:[?d] ?t, axis: list(int)): [] {ret_type} throws
      {where_statement}{{
      use SliceReductionOps;
      type opType = {ret_type};
      const (valid, axes) = validateNegativeAxes(axis, x.rank);
      if !valid {{
        throw new Error("Invalid axis value(s) '%?' in slicing reduction".format(axis));
      }} else {{
        const outShape = reducedShape(x.shape, axes);
        var ret = makeDistArray((...outShape), opType);
        forall (sliceDom, sliceIdx) in axisSlices(x.domain, axes)
          do ret[sliceIdx] = {func}Slice(x, sliceDom);
        return ret;
      }}
    }}"""
        ret += "\n"

    ret = clean_string(ret)

    return ret

def clean_string(my_code:str):
    return my_code.replace("\t", "  ").replace(r"\n\s*\n", "\n\n")

def generate_index_reduction_functions():
    ret = ""

    for func, ret_type, allowed_types in INDEX_FUNCS:

        where_statement = ""

        if len(allowed_types) > 0:
            where_statement += "where "
            where_statement += "||".join([f"(t=={tp})" for tp in allowed_types])

        ret += f"""
    @arkouda.registerCommand
    proc {func}All(const ref x:[?d] ?t): {ret_type} throws
    where (t != bigint) {{
      use SliceReductionOps;
      if d.rank == 1 {{
        return {func}Slice(x, d):{ret_type};
      }} else {{
        const ord = new orderer(x.shape);
        const ret = ord.indexToOrder({func}Slice(x, d)):{ret_type};
        return ret;
      }}
    }}

    @arkouda.registerCommand
    proc {func}(const ref x:[?d] ?t, axis: int): [] {ret_type} throws
      where (t != bigint) && (d.rank > 1) {{
      use SliceReductionOps;
      const axisArry = [axis];
      const outShape = reducedShape(x.shape, axisArry);
      var ret = makeDistArray((...outShape), {ret_type});
      forall sliceIdx in domOffAxis(d, axisArry) {{
        const sliceDom = domOnAxis(d, sliceIdx, axis);
        ret[sliceIdx] = {func}Slice(x, sliceDom)[axis]:{ret_type};
      }}
      return ret;
    }}
"""
        ret += "\n"

    ret = clean_string(ret)

    return ret


def main():

    outfile = "src/ReductionMsgFunctions.chpl"

    with open(outfile, "w") as text_file:
        text_file.write(generate_header())
        text_file.write(generate_reduction_functions())
        text_file.write(generate_index_reduction_functions())
        text_file.write(generate_footer())


if __name__ == "__main__":
    main()
