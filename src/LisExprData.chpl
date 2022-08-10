module LisExprData
{

    public use List;
    public use Map;
    
    type Symbol = string;

   /* allowed list value types */
    enum LVT {Lst, Sym, I, R};
    
    /* allowed value types in the eval */
    /* in a real scheme/lisp interpreter these two enums (LVT and VT) would be the same */
    enum VT {I, R};

    /* type: generic list values */
    type GenListValue = shared GenListValueClass;
    type BGenListValue = borrowed GenListValueClass;
    type ListValue = shared ListValueClass;
    type BListValue = borrowed ListValueClass;
    
    /* type: list of generic list values */
    type GenList = shared ListClass(GenListValue);
    type BGenlist = borrowed ListClass(GenListValue);
    
    /* type: generic values for eval */
    type GenValue = owned GenValueClass;
    type BGenValue = borrowed GenValueClass;
    type Value = owned ValueClass;
    type BValue = borrowed ValueClass;

    /*
      List Class wraps the standard List which is a record
      forwarding the interface to the class definition
      this gives me a more familiar feel to lists in python
    */
    class ListClass {
      type etype;
      forwarding var l: list(etype);
    }
    
    /* generic list value class def */
    class GenListValueClass
    {
        var lvt: LVT;
        
        /* initialize the list value type so we can test it at runtime */
        proc init(type lvtype) {
            if (lvtype == GenList)            {lvt = LVT.Lst;}
            if (lvtype == Symbol)             {lvt = LVT.Sym;}
            if (lvtype == int)                {lvt = LVT.I;}
            if (lvtype == real)               {lvt = LVT.R;}
        }
        
        /* cast to the GenListValue to borrowed ListValue(vtype) halt on failure */
        inline proc toListValue(type lvtype) {
            return try! this :BListValue(lvtype);
        }

        /* returns a copy of this... an owned GenListValue */
        proc copy(): GenListValue throws {
          select (this.lvt) {
            when (LVT.Lst) {
              return new ListValue(this.toListValue(GenList).lv);
            }
            when (LVT.Sym) {
              return new ListValue(this.toListValue(Symbol).lv);
            }
            when (LVT.I) {
              return new ListValue(this.toListValue(int).lv);
            }
            when (LVT.R) {
              return new ListValue(this.toListValue(real).lv);
            }
            otherwise {throw new owned Error("not implemented");}
          }
        }
    }
    
    /* concrete list value class def */
    class ListValueClass : GenListValueClass
    {
        type lvtype;
        var lv: lvtype;
        
        /* initialize the value and the vtype */
        proc init(val: ?vtype) {
            super.init(vtype);
            this.lvtype = vtype;
            this.lv = val;
            this.complete();
        }
        
    }

    
    /* generic value class */
    class GenValueClass
    {
        /* value type testable at runtime */
        var vt: VT;
    
        /* initialize the value type so we can test it at runtime */
        proc init(type vtype) {
            if (vtype == int)  {vt = VT.I;}
            if (vtype == real) {vt = VT.R;}
        }
        
        /* cast to the GenValue to borrowed Value(vtype) halt on failure */
        inline proc toValue(type vtype) {
            return try! this :BValue(vtype);
        }

        /* returns a copy of this... an owned GenValue */
        proc copy(): GenValue throws {
            select (this.vt) {
                when (VT.I) {return new Value(this.toValue(int).v);}
                when (VT.R) {return new Value(this.toValue(real).v);}
                otherwise { throw new owned Error("not implemented"); }
            }
        }
    }
    
    /* concrete value class */
    class ValueClass : GenValueClass
    {
        type vtype; // value type
        var v: vtype; // value
        
        /* initialize the value and the vtype */
        proc init(val: ?vtype) {
            super.init(vtype);
            this.vtype = vtype;
            this.v = val;
        }
    }

    //////////////////////////////////////////
    // operators over GenValue
    //////////////////////////////////////////
    
    inline operator +(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value(l.toValue(int).v + r.toValue(int).v);}
            when (VT.I, VT.R) {return new Value(l.toValue(int).v + r.toValue(real).v);}
            when (VT.R, VT.I) {return new Value(l.toValue(real).v + r.toValue(int).v);}
            when (VT.R, VT.R) {return new Value(l.toValue(real).v + r.toValue(real).v);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator -(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value(l.toValue(int).v - r.toValue(int).v);}
            when (VT.I, VT.R) {return new Value(l.toValue(int).v - r.toValue(real).v);}
            when (VT.R, VT.I) {return new Value(l.toValue(real).v - r.toValue(int).v);}
            when (VT.R, VT.R) {return new Value(l.toValue(real).v - r.toValue(real).v);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator *(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value(l.toValue(int).v * r.toValue(int).v);}
            when (VT.I, VT.R) {return new Value(l.toValue(int).v * r.toValue(real).v);}
            when (VT.R, VT.I) {return new Value(l.toValue(real).v * r.toValue(int).v);}
            when (VT.R, VT.R) {return new Value(l.toValue(real).v * r.toValue(real).v);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator <(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v < r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v < r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v < r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v < r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator >(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v > r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v > r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v > r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v > r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator <=(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v <= r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v <= r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v <= r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v <= r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator >=(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v >= r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v >= r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v >= r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v >= r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator ==(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v == r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v == r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v == r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v == r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline operator !=(l: BGenValue, r: BGenValue): GenValue throws {
        select (l.vt, r.vt) {
            when (VT.I, VT.I) {return new Value((l.toValue(int).v != r.toValue(int).v):int);}
            when (VT.I, VT.R) {return new Value((l.toValue(int).v != r.toValue(real).v):int);}
            when (VT.R, VT.I) {return new Value((l.toValue(real).v != r.toValue(int).v):int);}
            when (VT.R, VT.R) {return new Value((l.toValue(real).v != r.toValue(real).v):int);}
            otherwise {throw new owned Error("not implemented");}
        }
    }

    inline proc and(l: BGenValue, r: BGenValue): GenValue throws {
        return new Value((l && r):int);
    }

    inline proc or(l: BGenValue, r: BGenValue): GenValue throws {
        return new Value((l || r):int);
    }

    inline proc not(l: BGenValue): GenValue throws {
        return new Value((! isTrue(l)):int);
    }

    inline proc isTrue(gv: BGenValue): bool throws {
        select (gv.vt) {
            when (VT.I) {return (gv.toValue(int).v != 0);}
            when (VT.R) {return (gv.toValue(real).v != 0.0);}
            otherwise {throw new owned Error("not implemented");}
        }
    }
    
    /* environment is a dictionary of {string:GenValue} */
    class Env
    {
        /* map of Symbol to GenValue */
        var tab: map(Symbol, GenValue);

        /* add a new entry or set an entry to a new value */
        proc addEntry(name:string, val: ?t): BValue(t) throws {
            var entry = new Value(val);
            tab.addOrSet(name, entry);
            return tab.getBorrowed(name).toValue(t);
        }

        /* add a new entry or set an entry to a new value */
        proc addEntry(name:string, in entry: GenValue): BGenValue throws {
            tab.addOrSet(name, entry);
            return tab.getBorrowed(name);
        }

        /* lookup symbol and throw error if not found */
        proc lookup(name: string): BGenValue throws {
            if (!tab.contains(name)) {
              throw new owned Error("undefined symbol error " + name);
            }
            return tab.getBorrowed(name);
        }

        /* delete entry -- not sure if we need this */
        proc deleteEntry(name: string) {
            import IO.stdout;
            if (tab.contains(name)) {
                tab.remove(name);
            }
            else {
                writeln("unkown symbol ",name);
                try! stdout.flush();
            }
        }
    }


}
