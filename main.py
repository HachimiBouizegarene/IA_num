import tensorflow as tf
import numpy
import pygame
import math
from neuronalNetwork import NeuronalNetwork

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

pygame.init()
HEIGHT = 600
WIDTH = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
zoom  = 30
image_width = 28

#buttons
test_img = pygame.image.load('img/test-button.png')
train_img = pygame.image.load('img/train-button.png')
visual_img = pygame.image.load('img/visual-button.png')



#font
font = pygame.font.Font('RobotoCondensed-Bold.ttf', 32)


def toScreen (x, y):
    x/= zoom
    y/= zoom
    y*= -1
    x += 1
    y += 1
    x *= WIDTH * 0.5
    y *= HEIGHT * 0.5 
    return x, y

class Button():
    def __init__(self, x, y, image):
        self.width = image.get_width()
        self.image = image
        self.rect = self.image.get_rect()
        x , y = toScreen(x , y)
        self.rect.topleft= (x - self.width/2 , y)
    
    def draw(self):
        screen.blit(self.image, (self.rect.x, self.rect.y ))

    def focused(self):
        pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(pos)
            

test_btn = Button(0, -20, test_img)
train_btn = Button(-14, -20, train_img)
visual_btn = Button(14, -20, visual_img)

def toWidth(x):
    x/=zoom
    x*= WIDTH * 0.5
    return math.ceil(x + 0.01)

def toHeight(y):
    y/=zoom
    y*= HEIGHT * 0.5
    return math.ceil(y + 0.01)

def normalize(tab : list[int]):
    res = []
    for i in range(len(tab)):
        res.append( tab[i] / 255.0)
    return res

def drawPixel(x : int, y : int, c ):
    x , y = toScreen(x, y)
    width_ = toWidth(1)
    height_ = toHeight(1)
    pygame.draw.rect(screen, c, pygame.Rect(x , y, width_, height_))

def get_color(x):
    grayscale_value = x * 255
    return (grayscale_value, grayscale_value, grayscale_value)

def drawImage(pixels : list[float]) : 
    for y in range(image_width):
        for x in range(image_width):
            pixel = pixels[y * image_width + x]
            if x == 0 or y == 0 or x ==image_width- 1 or y == image_width -1: c = (149,81,73)
            else :
                c = get_color(pixel)
                if pixel < 0.1 : c = (92,186,160)
            drawPixel(x - 14, y - 5, c)



#neural Network

def toResTrain(n : int):
    res = []
    for i in range(10):
        res.append(0)
    res[n] = 1
    return res

def toRes(tab : list[float]):
    max = 0
    for i in range(len(tab)):
        max = i if tab[i] > tab[max] else max
    return max


nn = NeuronalNetwork(28 * 28, 32, 10)




def drawText(text : str, x, y):
    paddingY = 5
    paddingX = 40
    background_color = (226,233,192)
    text = font.render(text, True, (149,81,73), background_color)
    width = text.get_width()
    height = text.get_height()
    textRect = text.get_rect()
    x, y = toScreen(x, y)
    textRect.topleft = (x - width / 2, y)
    pygame.draw.rect(screen, background_color, pygame.Rect(x - width /2 - paddingX, y - paddingY, width + paddingX * 2, height + paddingY * 2))
    screen.blit(text, textRect)

def drawLoadBar(value : float):
    value_normalized = value / 100
    width = 30
    height = 3
    x, y = toScreen(0 - width / 2, -10)
    width = toWidth(width)
    height = toHeight(height)
    pygame.draw.rect(screen, (226,233,192), pygame.Rect(x-5, y-5, width + 10, height + 10))
    pygame.draw.rect(screen, (122,169,92), pygame.Rect(x, y, width * value_normalized, height))



train = False
current_image_train = 0
decate = 1500
training_count = 0 

visual = False
current_image_visual = 0
visual_tick_counter = 0

test = False
test_images_count = 300

current_image_test = 0
score = 0
score_100 = 0



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN :
            if (pygame.mouse.get_pressed()[0] == True) :
                if(test_btn.focused()) :
                    if(not train) :
                        visual = False
                        test = True
                
                if(train_btn.focused()) :
                    if(not test):
                        visual = False
                        train = True
                    
                if(visual_btn.focused()) :
                    if(not (train or test)):
                        visual = True
                

    tick = 60

    screen.fill((30,15,28))
    #training
    if(score_100 > 0):
        drawText(str(math.ceil(score_100)) + str("%"), 0, 10)

    if train :
        tick = 10000
        visual = False
        test = False
        image = normalize(train_images[current_image_train].reshape(-1))
        label = toResTrain(train_labels[current_image_train])
        nn.train(image, label)
        current_image_train += 1

        drawLoadBar((training_count / decate) * 100)

        training_count += 1
        if training_count >= decate : 
            training_count = 0
            train = False
        


    if(visual) : 
        if visual_tick_counter >= 60 :
            current_image_visual += 1
            visual_tick_counter = 0
        else : visual_tick_counter += 1
        image_visual = train_images[current_image_visual].reshape(-1)
        image_visual_nomalized = normalize(image_visual)
        label_guessed = toRes(nn.guess(image_visual_nomalized))
        drawText(str(label_guessed), 22, 10)
        drawImage(image_visual_nomalized)
    
 

    if(test):
        tick = 10000
        image_test = test_images[current_image_test].reshape(-1)
        image_test_nomalized = normalize(image_test)
        label_guessed = toRes(nn.guess(image_test_nomalized))
        label_wanted = test_labels[current_image_test]
        if(label_guessed == label_wanted) : score += 1
        current_image_test += 1

        drawLoadBar((current_image_test / test_images_count) * 100)

        if current_image_test >= test_images_count :
            score_100 = score / test_images_count * 100
            current_image_test = 0
            score = 0
            test = False

    test_btn.draw()
    visual_btn.draw()
    train_btn.draw()

    
    clock.tick(tick)
    pygame.display.flip()
    