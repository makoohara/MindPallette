import json
import spotipy
import webbrowser

# def search_and_play_song(search_keyword):
#     # Search for the Song.
#     search_results = spotifyObject.search(search_keyword, 1, 0, "track")
#     # Get required data from JSON response.
#     tracks_dict = search_results['tracks']
#     tracks_items = tracks_dict['items']
#     if tracks_items:
#         song_url = tracks_items[0]['external_urls']['spotify']
#         # Open the Song in Web Browser
#         webbrowser.open(song_url)
#         print('Song has opened in your browser.')
#     else:
#         print("No songs found for your search.")

# User credentials and Spotify setup
username = 'mako_o'
client_id = "e85b835ee3054763a2e8737070c34bf0"
client_secret = "c5fe80ec8be34582a207262d0f721879"
redirect_uri = "http://google.com/callback/"
scope = "user-read-playback-state,user-modify-playback-state"

# Create OAuth Object
oauth_object = spotipy.SpotifyOAuth(client_id, client_secret, redirect_uri, scope=scope)
# Create token
token_dict = oauth_object.get_access_token()
token = token_dict['access_token']
print(token)
# Create Spotify Object
spotifyObject = spotipy.Spotify(auth=token)

def search_and_play_song(search_keyword):
    # Search for the Song.
    search_results = spotifyObject.search(search_keyword, 1, 0, "track")
    # Get required data from JSON response.
    tracks_dict = search_results['tracks']
    tracks_items = tracks_dict['items']
    if tracks_items:
        return tracks_items[0]['external_urls']['spotify']
    else:
        return None

choice = input("Your Choice: ")
print(search_and_play_song(choice))
# # Create OAuth Object
# oauth_object = spotipy.SpotifyOAuth(client_id, client_secret, redirect_uri, scope=scope)
# # Create token
# token_dict = oauth_object.get_access_token()
# token = token_dict['access_token']
# # Create Spotify Object
# spotifyObject = spotipy.Spotify(auth=token)

# user = spotifyObject.current_user()
# # To print the response in readable format.
# print(json.dumps(user, sort_keys=True, indent=4))

# while True:
#     print("Welcome, "+ user['display_name'])
#     print("0 - Exit")
#     print("1 - Search for a Song")
#     choice = int(input("Your Choice: "))
#     if choice == 1:
#         search_keyword = input("Enter Song Name: ")
#         search_and_play_song(search_keyword)
#     elif choice == 0:
#         break
#     else:
#         print("Enter valid choice.")


