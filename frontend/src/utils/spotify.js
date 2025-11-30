export async function fetchPlaylistTracks(token, playlistUri) {
    if (!token || !playlistUri) return [];

    try {
        // Extract ID from URI (spotify:playlist:ID)
        const playlistId = playlistUri.split(':')[2];
        if (!playlistId) {
            console.warn('Invalid playlist URI format:', playlistUri);
            return [];
        }

        const response = await fetch(`https://api.spotify.com/v1/playlists/${playlistId}/tracks?limit=3`, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });

        if (!response.ok) {
            // Don't log 404 as error since some playlists may be unavailable
            if (response.status === 404) {
                console.warn(`Playlist not found (may be private or deleted): ${playlistId}`);
            } else {
                console.error(`Failed to fetch playlist tracks (${response.status}):`, response.statusText);
            }
            return [];
        }

        const data = await response.json();

        // Filter out null tracks (Spotify API sometimes returns null for unavailable tracks)
        const validItems = data.items.filter(item => item.track && item.track.name);
        
        if (validItems.length === 0) {
            console.warn(`Playlist ${playlistId} has no available tracks`);
            return [];
        }

        return validItems.map(item => ({
            title: item.track.name,
            artist: item.track.artists.map(a => a.name).join(', '),
            uri: item.track.uri
        }));
    } catch (error) {
        console.error('Error fetching playlist tracks:', error);
        return [];
    }
}
