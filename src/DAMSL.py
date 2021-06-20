class DAMSL:
    @staticmethod
    def da_to_cf(da):
        if da in ["O", "Q"]:
            return "Question"
        elif da in ["I", "A", "D", "W"]:
            return "Inform"
        elif da in ["E", "R", "S"]:
            return "Discussion"
        elif da in ["H", "T", "J", "V", "M", "F"]:
            return "Casual"
        else:
            print("da_to_cf:{}".format(da))
            assert da in ["O", "Q", "I", "A", "D", "W", "E", "R", "S", "F", "H", "T", "J", "V", "M"]
