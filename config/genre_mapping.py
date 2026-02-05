"""
Genre mapping configuration for dataset processing

Maps specific Spotify genres to 9 general categories used in the project.
"""

# Genre mapping to general categories
GENRE_GROUPS = {
    'Rock': [
        'rock', 'classic rock', 'album rock', 'hard rock',
        'soft rock', 'modern rock', 'alternative rock', 'indie rock',
        'progressive rock', 'symphonic rock', 'folk rock', 'post-grunge'
    ],

    'Metal': [
        'metal', 'alternative metal', 'nu metal', 'heavy metal',
        'death metal', 'black metal', 'metalcore'
    ],

    'Pop': [
        'pop', 'dance pop', 'pop dance', 'indie pop',
        'art pop', 'electropop', 'synth pop', 'pop rock',
        'teen pop', 'power pop'
    ],

    'Hip Hop': [
        'hip hop', 'rap', 'pop rap', 'trap', 'conscious hip hop',
        'gangster rap', 'southern hip hop', 'east coast hip hop',
        'west coast rap', 'emo rap'
    ],

    'Electronic': [
        'edm', 'electro house', 'progressive house', 'house',
        'techno', 'trance', 'dubstep', 'drum and bass',
        'indietronica', 'electro'
    ],

    'Classical': [
        'classical', 'baroque', 'classical era',
        'late romantic era', 'early romantic era',
        'post-romantic era', 'early music',
        'german romanticism', 'german baroque'
    ],

    'Jazz': [
        'jazz', 'vocal jazz', 'smooth jazz', 'contemporary jazz',
        'bebop', 'cool jazz', 'jazz fusion'
    ],

    'R&B': [
        'r&b', 'soul', 'neo soul', 'contemporary r&b',
        'funk', 'motown'
    ],

    'Country': [
        'country', 'contemporary country', 'country road',
        'classic country', 'country rock', 'bluegrass'
    ],

    'Latin': [
        'latin pop', 'reggaeton', 'latin', 'salsa',
        'bachata', 'regional mexican', 'banda'
    ],

    'Folk': [
        'folk', 'indie folk', 'folk rock', 'contemporary folk',
        'scottish folk', 'irish folk'
    ],

    'Alternative': [
        'alternative', 'alternative rock', 'indie',
        'permanent wave', 'new wave', 'post-punk'
    ],

    'Singer-Songwriter': [
        'singer-songwriter', 'lilith', 'mellow gold',
        'folk-pop'
    ],

    'Easy Listening': [
        'easy listening', 'adult standards', 'lounge',
        'bossa nova'
    ],

    'Soundtrack': [
        'soundtrack', 'orchestral soundtrack', 'score',
        'video game music'
    ]
}

# Final genres after balancing
# (The ones which have enough tracks to include in the dataset)
FINAL_GENRES = [
    'Rock',
    'Pop',
    'Classical',
    'Hip Hop',
    'Electronic',
    'Jazz',
    'Latin',
    'Country',
    'Soundtrack'
]

# Balancing parameters
MIN_TRACKS_PER_GENRE = 10_000
MAX_TRACKS_PER_GENRE = 40_000


def get_genre_mapping():
    """Returns the genre mapping dictionary"""
    return GENRE_GROUPS


def get_final_genres():
    """Returns list of genres in final dataset"""
    return FINAL_GENRES


def get_balancing_params():
    """Returns min/max tracks per genre"""
    return {
        'min_tracks': MIN_TRACKS_PER_GENRE,
        'max_tracks': MAX_TRACKS_PER_GENRE
    }