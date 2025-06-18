// This file is used only when generating docs -
// to satisfy `use RegistrationConfig` in ServerConfig.chpl.
// When building Arkouda, this module is found in `src/registry/Commands.chpl`.
module RegistrationConfig {

  /* This param contains verbatim the contents of `registration-config.json`
     that were used when building Arkouda. */
  param registrationConfigSpec: string;

  /* This param string contains a list of array dimensions requested in
     `registration-config.json`. For example: "1," or "1,2,3,". */
  param arrayDimensionsStr: string;

  /* This type indicates, implicitly, what array ranks were requensted in
     `registration-config.json`. For example: `(1*nothing,)` if just rank=1
     or `(1*nothing, 2*nothing, 3*nothing)` if it requested ranks 1, 2, and 3. */
  type arrayDimensionsTy;

  /* This is a tuple type each component of which is an array element type
     requested in `registration-config.json`. For example: `(int, bigint, real)`
     if `int`, `bigint`, and `real` were requested. */
  type arrayElementsTy;

}  // module RegistrationConfig
