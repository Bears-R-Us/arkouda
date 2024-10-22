FUNCS = [
    ["any", "bool", []],
    ["all", "bool", []],
    ["isSorted", "bool", []],
    ["isSortedLocally", "bool", []],
]


def generate_reduction_functions():

    ret = """module ReductionMsgFunctions
{
    use BigInteger;
    use List;
    use AryUtil;
    use ReductionMsg;
    use SliceReductionOps;
"""

    for func, ret_type, allowed_types in FUNCS:

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
        ret += "\n\n"

    ret = ret.replace("\t", "  ")
    ret = ret
    ret += "\n}"

    return ret


def main():

    outfile = "src/ReductionMsgFunctions.chpl"

    with open(outfile, "w") as text_file:
        text_file.write(generate_reduction_functions())


if __name__ == "__main__":
    main()
