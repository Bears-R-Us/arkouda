import base64
import cloudpickle #type: ignore

import arkouda as ak

__all__ = ['for_each']


def for_each(pda_in, functor, inplace=False):
    s = cloudpickle.dumps(functor)

    enc_s = base64.b64encode(s)
    u_enc_s = enc_s.decode('UTF-8')

    server_msg = ak.client.generic_msg(
        "python1D",
        {"a": pda_in, "pickle": u_enc_s, "inplace": inplace})

    if not inplace:
        return ak.create_pdarray(server_msg)

    return pda_in
