export async function fetchPlaylistTracks(token, playlistUri) {
    if (!token || !playlistUri) return [];

    try {
        // Extract ID from URI (spotify:playlist:ID)
        const playlistId = playlistUri.split(':')[2];

        const response = await fetch(`https://api.spotify.com/v1/playlists/${playlistId}/tracks?limit=3`, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });

        if (!response.ok) {
            console.error('Failed to fetch playlist tracks:', response.statusText);
            return [];
        }

        const data = await response.json();

        return data.items.map(item => ({
            title: item.track.name,
            artist: item.track.artists.map(a => a.name).join(', '),
            uri: item.track.uri
        }));
    } catch (error) {
        console.error('Error fetching playlist tracks:', error);
        return [];
    }
}
