class Splitting_level_node:

    def __init__(self, number_of_instances: int, link="", parent=None):
        self.number_of_instances = number_of_instances
        self.link = link
        self.children = []
        self.return_value = 0
        if parent:
            parent.children.append(self)

    def __str__(self):
        l = "" if self.link == "" else f", l: {self.link}"
        r = "" if self.return_value == 0 else f", r: {self.return_value} "
        return f"#inst: {self.number_of_instances}{l}{r}"
