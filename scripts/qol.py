import pygame

def play_done_sound(path="/path/to/song.mp3"):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass