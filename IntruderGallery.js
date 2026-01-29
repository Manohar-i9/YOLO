import React, { useState, useEffect } from "react";
import { View, Text, FlatList, Image, StyleSheet, TextInput, TouchableOpacity, Modal, Button, Alert, TouchableWithoutFeedback } from "react-native";
import { getStorage, ref, listAll, getDownloadURL } from "firebase/storage"; // Import Firebase Storage functions
import { getFirestore, collection, onSnapshot } from "firebase/firestore";  // Correct Firestore import

export default function IntruderGallery() {
    const [images, setImages] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortState, setSortState] = useState('name-asc');
    const [modalVisible, setModalVisible] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);
    const [renameModalVisible, setRenameModalVisible] = useState(false);
    const [newName, setNewName] = useState('');
    const [selectedPhotoId, setSelectedPhotoId] = useState(null);
    const [isDropdownVisible, setIsDropdownVisible] = useState(false);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 });
    const [loading, setLoading] = useState(false);

    // Firebase services should be used directly here
    const storage = getStorage();  // This now works after initialization
    const firestore = getFirestore();  // This now works after initialization

    const fetchImages = async () => {
        setLoading(true);
        try {
            const reference = ref(storage, 'intruders'); 
            const result = await listAll(reference);
    
            const imageUrls = await Promise.all(result.items.map(async (item) => {
                const url = await getDownloadURL(item);
                return { id: item.name, uri: url, name: item.name, timestamp: new Date().toISOString() };
            }));
    
            setImages(imageUrls.reverse());
        } catch (error) {
            console.error("Error fetching images from Firebase Storage:", error);
        }
        setLoading(false);
    };

    useEffect(() => {
        fetchImages();  // Initial fetch

        // Auto-refresh every 10 seconds
        const interval = setInterval(fetchImages, 3000);

        // Firestore live updates
        const unsubscribe = onSnapshot(collection(firestore, 'image_metadata'), (snapshot) => {
            const newImages = snapshot.docs.map(doc => doc.data());
            setImages(newImages);
        });

        return () => {
            clearInterval(interval);
            unsubscribe();
        };
    }, []);

    const handleSort = () => {
        let sortedImages = [...images];
        if (sortState === 'name-asc') {
            sortedImages.sort((a, b) => a.name.localeCompare(b.name));
            setSortState('name-desc');
        } else if (sortState === 'name-desc') {
            sortedImages.sort((a, b) => b.name.localeCompare(a.name));
            setSortState('date-asc');
        } else if (sortState === 'date-asc') {
            sortedImages.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            setSortState('date-desc');
        } else {
            sortedImages.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            setSortState('name-asc');
        }
        setImages(sortedImages);
    };

    const handleSearch = (text) => setSearchQuery(text);

    const openModal = (uri) => {
        setSelectedImage(uri);
        setModalVisible(true);
    };

    const handleLongPress = (id, event) => {
        setSelectedPhotoId(id);
        setIsDropdownVisible(true);
        const { pageX, pageY } = event.nativeEvent;
        setDropdownPosition({ top: pageY, left: pageX });
    };

    const handleRename = () => {
        setImages(images.map(image => image.id === selectedPhotoId ? { ...image, name: newName } : image));
        setRenameModalVisible(false);
        setIsDropdownVisible(false);
        setNewName('');
    };

    const handleDelete = () => {
        setImages(images.filter(image => image.id !== selectedPhotoId));
        setIsDropdownVisible(false);
    };

    const handleDownload = async (imageName) => {
        Alert.alert('Mock Download', 'This feature is simulated in development.');
        setIsDropdownVisible(false);
    };

    const closeDropdown = () => {
        setIsDropdownVisible(false);
        setNewName('');
    };

    const filteredImages = images.filter(image =>
        image.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <TouchableWithoutFeedback onPress={closeDropdown}>
            <View style={styles.container}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={handleSort} style={styles.sortButton}>
                        <Text style={styles.sortText}>Sort</Text>
                    </TouchableOpacity>
                    <TextInput
                        style={styles.searchInput}
                        placeholder="Search by name"
                        value={searchQuery}
                        onChangeText={handleSearch}
                    />
                </View>

                {/* Gallery Grid */}
                <FlatList
                    data={filteredImages}
                    keyExtractor={item => item.id}
                    numColumns={2}
                    renderItem={({ item }) => (
                        <TouchableOpacity
                            style={styles.imageCard}
                            onPress={() => openModal(item.uri)}
                            onLongPress={(e) => handleLongPress(item.id, e)}
                        >
                            <Image source={item.uri} style={styles.image} />
                            <Text style={styles.name}>{item.name}</Text>
                            <Text style={styles.timestamp}>{item.timestamp}</Text>
                        </TouchableOpacity>
                    )}
                />

                {/* Fullscreen Image Modal */}
                <Modal visible={modalVisible} transparent onRequestClose={() => setModalVisible(false)}>
                    <View style={styles.modalContainer}>
                        <Image source={selectedImage} style={styles.fullImage} />
                        <Button title="Close" onPress={() => setModalVisible(false)} />
                    </View>
                </Modal>

                {/* Rename Modal */}
                <Modal visible={renameModalVisible} transparent animationType="slide">
                    <View style={styles.modalContainer}>
                        <TextInput
                            placeholder="Enter new name"
                            placeholderTextColor="white"
                            style={styles.renameInput}
                            value={newName}
                            onChangeText={setNewName}
                        />
                        <View style={styles.buttonContainer}>
                            <Button title="Rename" onPress={handleRename} />
                        </View>
                        <View style={styles.buttonContainer}>
                            <Button title="Cancel" onPress={() => setRenameModalVisible(false)} />
                        </View>
                    </View>
                </Modal>

                {/* Dropdown Menu */}
                {isDropdownVisible && (
                    <View style={[styles.dropdown, { top: dropdownPosition.top, left: dropdownPosition.left }]}>
                        <TouchableOpacity onPress={() => setRenameModalVisible(true)}>
                            <Text style={styles.dropdownItem}>Rename</Text>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={handleDelete}>
                            <Text style={styles.dropdownItem}>Delete</Text>
                        </TouchableOpacity>
                        <TouchableOpacity onPress={handleDownload}>
                            <Text style={styles.dropdownItem}>Download</Text>
                        </TouchableOpacity>
                    </View>
                )}
            </View>
        </TouchableWithoutFeedback>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, padding: 10, backgroundColor: "#f5f5f5" },
    header: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
    sortButton: { backgroundColor: '#007BFF', paddingTop: 10, paddingBottom: 10, paddingLeft: 20, paddingRight: 20, borderRadius: 5 },
    sortText: { color: '#fff' },
    searchInput: { borderWidth: 1, borderRadius: 5, padding: 8, flex: 1, marginLeft: 10 },
    imageCard: { flex: 1, margin: 5, backgroundColor: '#fff', padding: 5, borderRadius: 8, elevation: 3 },
    image: { width: '100%', height: 120, borderRadius: 8 },
    name: { textAlign: 'center', marginTop: 5, fontWeight: 'bold' },
    timestamp: { textAlign: 'center', fontSize: 12, color: '#555' },
    modalContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.7)' },
    fullImage: { width: '90%', height: '80%', resizeMode: 'contain' },
    renameInput: { borderWidth: 1, borderRadius: 5, borderColor: 'white', width: 200, padding: 8, marginBottom: 10 },
    buttonContainer: { marginVertical: 10, width: '40%' },
    dropdown: { position: 'absolute', backgroundColor: 'white', padding: 10, borderRadius: 5, elevation: 5 },
    dropdownItem: { padding: 10 },
});
