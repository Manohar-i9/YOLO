import React, { useEffect } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Ionicons } from "@expo/vector-icons";
import LoginScreen from "./screens/LoginScreen";
import SignupScreen from "./screens/SignupScreen";
import HomeScreen from "./screens/HomeScreen";
import IntruderGallery from "./screens/IntruderGallery";
import LiveFeed from "./screens/LiveFeed";
import Alerts from "./screens/Alerts";
import Settings from "./screens/Settings";
import messaging from '@react-native-firebase/messaging';
import { Alert } from 'react-native';

// App.js Code ...
const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

// Function to send the FCM token to your backend
const sendTokenToBackend = async (token) => {
  try {
    const response = await axios.post('http://10.71.61.3:8000/api/store-token', {
      fcm_token: token,
      user_id: userId,
    });
    console.log("FCM Token sent to backend:", response.data);
  } catch (error) {
    console.error("Error sending FCM Token to backend:", error);
  }
};

// ✅ Full App with Bottom Tabs
function FullAppNavigation() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ color, size }) => {
          let iconName;
          if (route.name === "Home") iconName = "home";
          else if (route.name === "Gallery") iconName = "images";
          else if (route.name === "Live Feed") iconName = "videocam";
          else if (route.name === "Alerts") iconName = "notifications";
          else if (route.name === "Settings") iconName = "settings";
          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: "#007AFF",
        tabBarInactiveTintColor: "gray",
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Gallery" component={IntruderGallery} />
      <Tab.Screen name="Live Feed" component={LiveFeed} />
      <Tab.Screen name="Alerts" component={Alerts} />
      <Tab.Screen name="Settings" component={Settings} />
    </Tab.Navigator>
  );
}

// ✅ Handles Authentication Flow Automatically
function AuthFlow() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return null; // Prevent flickering
  }

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {isAuthenticated ? (
        <Stack.Screen name="FullApp" component={FullAppNavigation} />
      ) : (
        <>
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="Signup" component={SignupScreen} />
        </>
      )}
    </Stack.Navigator>
  );
}

export default function App() {
  

  return (
    <AuthProvider>
      <NavigationContainer>
        <AuthFlow />
      </NavigationContainer>
    </AuthProvider>
  );
}