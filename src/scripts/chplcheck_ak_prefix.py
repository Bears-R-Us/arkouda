import re

import chapel

# simple regex for “lowerCamelCase”: starts with lowercase, then zero or more groups
# of an uppercase letter + letters/digits, with no underscores
_camel_re = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)*$")


def rules(driver):
    @driver.basic_rule(chapel.Function)
    def CamelCaseFunctionsAkPrefixAllowed(context, node):
        name = node.name()
        # 1) allow our special prefix
        if name.startswith("ak_"):
            # strip the prefix and underscore before checking
            name = name[3:]
        # 2) enforce camelCase on the remainder
        return bool(_camel_re.match(name))
