class Demangler:

    def __init__(self):
        self.count = 0

    def VCDemangle(self, mangled_str, get_class=False):
        pretty_name = mangled_str
        classes = []

        if not mangled_str:
            return mangled_str
        
        mangled_str_len = len(mangled_str)

        if mangled_str_len < 3:
            return mangled_str
        
        if mangled_str[0] == '?' and mangled_str[1] != '?':
            at_index = mangled_str.find('@', 0)
            if at_index != -1:
                pretty_name = mangled_str[1:at_index]
                
            if get_class:
                curr_index = at_index + 1
                
                while curr_index < mangled_str_len:
                    if mangled_str[curr_index] != '@':
                        end_class = mangled_str.find('@', curr_index)
                        
                        if end_class != -1:
                            if end_class + 2 >= mangled_str_len:
                                break                        
                            if mangled_str[end_class + 2] == '@':
                                break

                            new_class = mangled_str[curr_index:end_class]
                            classes.append(new_class)

                        curr_index = end_class + 1
                    else:
                        break

            if get_class:
                pretty_class = ""
                classes.reverse()
                for cls in classes:
                    pretty_class += cls + "::"

                pretty_name = pretty_class + pretty_name

        return pretty_name        
