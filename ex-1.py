from random import random

import pandas as pd

comments = [
    "Amazing movie, really loved it!",
    "Terrible film, waste of time.",
    "The acting was fantastic, but the plot was weak.",
    "Great visuals but the story was a bit slow.",
    "I would watch it again, highly recommended!",
    "Not my cup of tea, but some might like it.",
    "A masterpiece! Stunning performances.",
    "Predictable and boring.",
    "Absolutely thrilling, kept me on the edge of my seat.",
    "Disappointed, expected much better.",
    "Unique storyline, very refreshing.",
    "Poorly written, felt rushed.",
    "Loved the cinematography, breathtaking!",
    "An okay film, nothing special.",
    "Horrible acting, couldn't finish it."
]

# Classifications: Positive, Neutral, Negative
classifications = ["Positive", "Neutral", "Negative"]

num_samples = 50
data = {
    "Comment": [random.choice(comments) for _ in range(num_samples)],
    "Classification": [random.choice(classifications) for _ in range(num_samples)]
}

comments_df = pd.DataFrame(data)


