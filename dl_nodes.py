

class String2ListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",  {"forceInput": True}),
                "separator": ("STRING", {"default": "/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    #OUTPUT_IS_LIST = (True, )
    FUNCTION = "perform_split_string"
    CATEGORY = "DLNodes"

    def perform_split_string(self, text, separator):
        str_list = text.split(separator)
        ret_arr = []
        for s in str_list:
            ret_arr += [s]
            
        return (ret_arr, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "String2List": String2ListNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "String2List": "String2List"
}
