"""
This class is just to create an easy internal representation of each song.
"""

class Song():
    def __init__(self, args):
        self.name = args[0]
        self.artist = args[1]
        self.album = args[2]
        self.lyrics = args[3]
        self.mgenre = args[4]
        self.sgenres = args[5]
        self.pplyrics = args[6]
        #Reminder that this property should be a list/dict representing the three available vector rep methods
        self.vector = args[7]
        self.id = args[8]

    def __str__(self):
        return f"{self.name} by {self.artist}. TEST: {self.lyrics}"