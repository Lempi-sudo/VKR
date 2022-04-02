class WaterMarkWrong(Exception):
    def __init__(self, text , right_size , Hwrong_size ,  Vwrong_size):
        self.txt = text
        self.right_size=right_size
        self.Hwrong_size=Hwrong_size
        self.Vwrong_size = Vwrong_size

    def __str__(self):
        message = self.txt + f". Необходимый размер {self.right_size}x{self.right_size} , а размер данной картинки {self.Hwrong_size}x{self.Vwrong_size} "
        return message
