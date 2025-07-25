from PIL import ImageFont
import visualkeras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def build_model():
    filters = 128
    model = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 3)),

        layers.Conv2D(filters, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//4, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),

        layers.Dense(32, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.25),
        # layers.Dense(dense2, activation="relu"),

        layers.Dense(9, activation='softmax', dtype='float32')
    ])
    return model

model = build_model()
model.summary()
"""
from graphviz import Digraph

# Create a simple left-to-right diagram of the three layers
dot = Digraph(format='png')
dot.attr(rankdir='LR', dpi='300')

# Define the three layers with details
dot.node('conv', 'Conv2D\nfilters=128\nkernel=(2×2)\nactivation=relu\npadding=same', shape='box')
dot.node('bn', 'BatchNormalization', shape='box')
dot.node('pool', 'MaxPooling2D\npool_size=(2×2)', shape='box')

# Connect the nodes
dot.edge('conv', 'bn')
dot.edge('bn', 'pool')

# Render and display
output_path = dot.render(filename='block_detailed', directory='diagrams', cleanup=True)
"""

from graphviz import Digraph
import os

# Filters configuration
filters = 256
filters_list = [filters, filters//2, filters//2, filters//4, filters//2, filters//2, filters, filters]

# Create graph with neato for manual positioning
dot = Digraph(format='png', engine='neato')
dot.graph_attr.update(dpi='300')
dot.attr(overlap='false', splines='true')

# Node positions for a roughly square layout
positions = {
    'input': '-1.0,3!',
    'block1': '-1.0, 2.5!',
    'block2': '0.5, 2.5!',
    'block3': '-1.0, 2!',
    'block4': '0.5, 2!',
    'block5': '-1.0, 1.5!',
    'block6': '0.5, 1.5!',
    'block7': '-1.0, 1.0!',
    'block8': '0.5, 1.0!',
    #'flatten': '0.5, 1.0!',
    #'dense1': '-1.0, 0.5!',
    'globalaverage': '-1.0, 0.5!',
    'dense':  '0.0, 0.5!',
    'output': '0.5, 0.5!'
}

# Input node
dot.node('input', 'Input\n256×256×3', shape='box', pos=positions['input'])

# Block nodes
for i in range(8):
    label = f'Block {i+1}\nConv2D({filters_list[i]})→BatchNorm→MaxPool'
    dot.node(f'block{i+1}', label, shape='box', pos=positions[f'block{i+1}'])

# dot.node('flatten', 'Flatten', shape='box', pos=positions['flatten'])
# dot.node('dense1', 'Dense(32)', shape='box', pos=positions['dense1'])
dot.node('globalaverage', 'GlobalAverage', shape='box', pos=positions['globalaverage'])
dot.node('dense', 'Dense(9)', shape='box', pos=positions['dense'])
dot.node('output', 'Output', shape='box', pos=positions['output'])


# Connect edges in sequence
sequence = ['input'] + [f'block{i+1}' for i in range(8)] + ['globalaverage'] + ['dense'] + ['output']
for src, dst in zip(sequence, sequence[1:]):
    dot.edge(src, dst)

# Ensure output directory exists
os.makedirs('diagrams', exist_ok=True)

# Render and save
output_path = dot.render(filename='model_diagram_grid', directory='diagrams', cleanup=True)

# Feedback
print(f"Diagram saved to {output_path}")


"""
plot_model(
    model,
    to_file='model_diagram.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB'
)
print("Saved model_diagram.png")


font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font = ImageFont.truetype(font_path, 32)

visualkeras.layered_view(model, legend = True, font = font, to_file='output.png').show() # write and show
"""