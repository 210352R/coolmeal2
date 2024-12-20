import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ChatbotScreen extends StatefulWidget {
  const ChatbotScreen({super.key});

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  final TextEditingController _controller = TextEditingController();

  final List<String> _messages = ['Hi !'];
  final List<String> _responses = ["Hi !! I am food related chat bot . "];
  // For set circular process indicater
  bool _isLoading = false;
  var historyMessages = [];
  String sessionId = '';

  // get current user id from firebase
  final String user = FirebaseAuth.instance.currentUser!.uid;

  // Function to make a POST request
  Future<void> _postRequest() async {
    String url = 'http://13.60.182.147/getSession?user_id=$user';

    try {
      // Replace the body map with your desired request payload
      Map<String, String> body = {
        'key1': 'value1',
        'key2': 'value2',
      };

      // Making the POST request
      final response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        // Successful response, extract the data
        final data = jsonDecode(response.body);

        setState(() {
          sessionId =
              data['session_id']; // Assuming the response contains 'userId'
        });
        print('Session ID: $sessionId');
      } else {
        // Handle error response
        print('Failed to get data: ${response.statusCode}');
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  // Method to call the API
  Future<void> fetchChatHistory() async {
    print('Fetching chat history...');
    final url =
        'http://13.60.182.147/gethistory?user_id=$user'; // Replace with your actual API URL

    try {
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        print("Response: ${response.body}");
        // If the server returns a successful response
        final data = jsonDecode(response.body);

        // Parse the "history" array
        List history = data['history'];
        setState(() {
          historyMessages = history;
        });
        print(historyMessages);
      } else {
        // Handle error
        print('Failed to load chat history');
      }
    } catch (e) {
      print('Error: $e');
    }
  }

  //print user id
  @override
  void initState() {
    super.initState();
    print('User ID: $user');
    _postRequest();
    fetchChatHistory();
  }

  @override
  void dispose() {
    // Call your method here when leaving the tab
    _triggerMethodOnLeave();
    super.dispose();
  }

  void _triggerMethodOnLeave() async {
    print('Leaving ChatbotScreen tab');
    if (_messages.length > 1) {
      await _insertSessionData();
    }
  }

  Future<void> _insertSessionData() async {
    print("Caalling insert session data");
    final String url =
        'http://13.60.182.147/insertSessionData?session_id=$sessionId';

    try {
      final response = await http.post(Uri.parse(url));

      if (response.statusCode == 200) {
        print('Session data inserted successfully.');
      } else {
        print('Failed to insert session data: ${response.statusCode}');
      }
    } catch (e) {
      print('Error while inserting session data: $e');
    }
  }

  Future<void> _sendMessage() async {
    final userMessage = _controller.text;

    if (userMessage.isEmpty) return;

    // Add the user's message to the list
    setState(() {
      _messages.add(userMessage);
      _controller.clear();
      _isLoading = true; // Show the loading indicator
    });

    final url = Uri.parse('http://13.60.182.147/chat');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        "query": userMessage, // Correctly formatted JSON
        "session_id": sessionId // Single set of curly braces for the map
      }),
    );

    if (response.statusCode == 200) {
      // Parse the response body
      final responseBody = json.decode(response.body);

      setState(() {
        _responses.add(responseBody['response']);
        _isLoading = false; // Hide the loading indicator
      });
    } else {
      // Handle error response
      setState(() {
        _responses.add('Error: Could not retrieve response');
        _isLoading = false; // Hide the loading indicator
      });
    }
  }

  // //Dummy history data
  // final List<String> _historyMessages = [
  //   "User: What is Flutter?",
  //   "AI: Flutter is an open-source UI software development kit.",
  //   "User: How does state management work?",
  //   "AI: There are various methods, like Provider, Riverpod, etc."
  // ];

  void printPop() {
    print('Popping the screen');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'CoolBOT',
          style: TextStyle(
            color: Color.fromARGB(255, 0, 193, 39),
            fontSize: 30,
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(
              Icons.notifications,
              color: Color.fromARGB(255, 2, 2, 2),
              size: 30,
            ),
            onPressed: () {},
          ),
        ],
      ),
      drawer: Drawer(
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 0),
          color: Colors.white, // Background color of the drawer
          child: ListView.builder(
            itemCount: historyMessages.length,
            itemBuilder: (context, index) {
              return Container(
                margin: const EdgeInsets.symmetric(
                    vertical: 8, horizontal: 10), // Margin around each item
                decoration: BoxDecoration(
                  color: const Color.fromARGB(
                      255, 238, 245, 255), // Light blue background color
                  borderRadius: BorderRadius.circular(12), // Rounded corners
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.2), // Shadow color
                      spreadRadius: 2, // Spread radius of the shadow
                      blurRadius: 5, // Blur radius of the shadow
                      offset: const Offset(0, 3), // Shadow offset
                    ),
                  ],
                ),
                child: ListTile(
                  leading: const Icon(Icons.message,
                      color: Colors.teal), // Icon color
                  title: Text(
                    historyMessages[index][1],
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w400, // Medium font weight
                      fontFamily: 'Roboto',
                      color: Colors.black87, // Text color
                    ),
                  ),
                  // Optionally, you can add a subtitle for additional information
                  // subtitle: Text('Subtitle here', style: TextStyle(fontSize: 14)),
                  onTap: () {
                    // When tapped, update the messages and responses in the chat view
                    setState(() {
                      _messages.clear(); // Clear current messages
                      _responses.clear(); // Clear current responses

                      // Assuming `historyMessages[index]` contains a list of messages and responses
                      List history = historyMessages[index];

                      // Loop through each item in the history and add them to the chat
                      for (var chat in history) {
                        if (chat[0]['type'] == 'user') {
                          _messages.add(chat['message']); // Add user message
                        } else if (chat[0]['type'] == 'bot') {
                          _responses.add(chat['message']); // Add bot response
                        }
                      }
                    });

                    // Close the drawer after selecting
                    Navigator.of(context).pop();
                  },
                ),
              );
            },
          ),
        ),
      ),
      body: Stack(
        children: [
          Column(
            children: [
              // Add image
              const Image(
                image: AssetImage('assets/images/chatbot_page.png'),
                width: 200,
                height: 200,
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: _messages.length,
                  itemBuilder: (context, index) {
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment
                          .stretch, // Allows full-width alignment
                      children: [
                        // User's message aligned to the right
                        Align(
                          alignment: Alignment.centerRight,
                          child: Container(
                            margin: const EdgeInsets.symmetric(
                                vertical: 10, horizontal: 10),
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: const Color.fromARGB(255, 124, 246, 134),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Text(
                              _messages[index],
                              style: const TextStyle(color: Colors.black),
                            ),
                          ),
                        ),
                        // AI response aligned to the left if available
                        if (index < _responses.length)
                          Align(
                            alignment: Alignment.centerLeft,
                            child: Container(
                              margin: const EdgeInsets.symmetric(
                                  vertical: 10, horizontal: 10),
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: const Color.fromARGB(255, 211, 232, 231),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                _responses[index],
                                style: const TextStyle(color: Colors.black87),
                              ),
                            ),
                          ),
                      ],
                    );
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _controller,
                        decoration: InputDecoration(
                          hintText: 'Type here',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(25.0),
                          ),
                          filled: true,
                          fillColor: Colors.grey[200],
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    GestureDetector(
                      onTap: _sendMessage,
                      child: Container(
                        padding: const EdgeInsets.all(15),
                        decoration: const BoxDecoration(
                          color: Colors.teal,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.send,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          // Display the loading indicator when _isLoading is true
          if (_isLoading)
            const Center(
              child: CircularProgressIndicator(),
            ),
        ],
      ),
    );
  }
}
