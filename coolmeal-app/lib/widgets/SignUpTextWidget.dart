import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SignUpTextWidget extends StatelessWidget {
  const SignUpTextWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: RichText(
        text: TextSpan(
          text: "Don't have an account? ",
          style: GoogleFonts.poppins(
            textStyle: const TextStyle(color: Colors.black),
          ),
          children: <TextSpan>[
            TextSpan(
              text: 'Sign Up',
              style: GoogleFonts.poppins(
                textStyle: const TextStyle(
                  color: Colors.lightGreen,
                  fontWeight: FontWeight.w600,
                ),
              ),
              // Add onTap for sign-up functionality
            ),
          ],
        ),
      ),
    );
  }
}
