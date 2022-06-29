def objtypedec(orig_cls):
    orig_cls.objtype = orig_cls.__name__
    return orig_cls
