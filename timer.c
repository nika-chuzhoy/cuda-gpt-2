#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const size_t seed = 123;

float time(const char *prompt, char *type) {
    FILE *fp;
    char output[1024]; // Buffer to store the output
    char response[1024]; // AI response
    double timing;

    // Assemble the prompt
    size_t length = snprintf(NULL, 0, "make %s_seed seed=%zu prompt=\"%s\"", type, seed, prompt) + 1; 
    char* command = (char*)malloc(length);
    if (command == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Format the string
    snprintf(command, length, "make %s_seed seed=%zu prompt=\"%s\"", type, seed, prompt);

    // Run the command and open a pipe to read its output
    fp = popen(command, "r");
    if (fp == NULL) {
        perror("Failed to run command");
        exit(EXIT_FAILURE);
    }

    // Read the output of the command until the end
    while (fgets(output, sizeof(output), fp) != NULL) {
        // Get and print the AI response
        //printf(output);
        //printf("\nCHUNK\n");
        char *ai_response = strstr(output, "AI: ");
        if (ai_response != NULL) {
            strcpy(response, ai_response + 4); // Copy the AI response, skipping "AI: "
            printf("%s response: %s", type, response);
        }
        // Check if the line contains the timing information
        if (strstr(output, "----Seconds to respond:") != NULL) {
            // Extract the timing number
            sscanf(output, "----Seconds to respond: %lf", &timing);
        }
    }

    // Close the pipe
    pclose(fp);

    // Print the extracted timing number
    printf("%s timing: %.6lf\n", type, timing);

    free(command);

    return timing;
}

int main() {
    // Array of prompt strings
    const size_t num_prompts = 3; // You can add more prompts
    const char *prompts[] = {"I don't like to code", "I like tea", "Testing"};

    // Variables to store total times for CPU and GPU
    float total_cpu_time = 0.0f;
    float total_gpu_time = 0.0f;

    printf("\ntimer program is running. this may take a few minutes...\n\n");

    // Loop through the list of prompts
    for (int i = 0; i < num_prompts; i++) {
        const char *prompt = prompts[i];
        printf("prompt: %s\n", prompt);
        
        // Call time_cpu() and time_gpu() functions with the current prompt
        float cpu_time = time(prompt, "cpu");
        float gpu_time = time(prompt, "gpu");

        // Accumulate total times for CPU and GPU
        total_cpu_time += cpu_time;
        total_gpu_time += gpu_time;
    }

    // Print the total times for CPU and GPU
    printf("\n");
    printf("total cpu time: %.6f\n", total_cpu_time);
    printf("total gpu time: %.6f\n", total_gpu_time);

    return 0;
}