import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Switch,
  StyleSheet,
  Alert,
} from "react-native";
import { useAuth } from "../context/AuthContext";
import { getAuth, updatePassword } from "firebase/auth";
import { Ionicons } from "react-native-vector-icons";

export default function SettingsScreen({ navigation }) {
  const { logout } = useAuth();
  const [darkMode, setDarkMode] = useState(false);

  const auth = getAuth();
  const user = auth.currentUser; // Get logged-in user

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const handleChangePassword = () => {
    Alert.prompt(
      "Change Password",
      "Enter your new password",
      [
        {
          text: "Cancel",
          style: "cancel",
        },
        {
          text: "Update",
          onPress: (newPassword) => {
            updatePassword(user, newPassword)
              .then(() =>
                Alert.alert("Success", "Password updated successfully!")
              )
              .catch((error) => Alert.alert("Error", error.message));
          },
        },
      ],
      "secure-text"
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>⚙️ Settings</Text>

      {/* Dark Mode Toggle */}
      <View style={styles.optionRow}>
        <Ionicons name="moon" size={24} color="white" />
        <Text style={styles.optionText}>Dark Mode</Text>
        <Switch value={darkMode} onValueChange={toggleDarkMode} />
      </View>

      {/* Change Password */}
      <TouchableOpacity style={styles.optionRow} onPress={handleChangePassword}>
        <Ionicons name="lock-closed" size={24} color="white" />
        <Text style={styles.optionText}>Change Password</Text>
      </TouchableOpacity>

      {/* Logout Button */}
      <TouchableOpacity style={styles.logoutButton} onPress={() => logout()}>
        <Ionicons name="log-out-outline" size={24} color="white" />
        <Text style={styles.logoutText}>Logout</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: "#121212" },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
    marginBottom: 20,
  },
  optionRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#222",
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  optionText: { flex: 1, color: "white", fontSize: 16, marginLeft: 10 },
  logoutButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#D32F2F",
    padding: 15,
    borderRadius: 10,
    marginTop: 20,
  },
  logoutText: { color: "white", fontSize: 16, marginLeft: 10 },
});
