//
//  GeminiService.swift
//  LLMFarm
//
//  Gemini API integration for dual-model conversational AI
//  This service handles HTTP requests to Google's Gemini API and parses
//  streaming thoughts in [bt]...[et] format for use with local SmolLM model
//

import Foundation

class GeminiService: ObservableObject {
    
    private let apiKey = "YOUR_GEMINI_API_KEY" // TODO: Replace with actual key
    private let baseURL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent"
    
    // Cache the template so we don't read the file every time
    private var promptTemplate: String?
    
    func streamThoughts(transcript: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    print("Building prompt...")
                    let prompt = buildPrompt(transcript: transcript)
                    print("Prompt built successfully")
                    
                    print("Making HTTP request...")
                    let request = buildRequest(prompt: prompt)
                    print("Request created, sending...")
                    
                    let (data, response) = try await URLSession.shared.data(for: request)
                    print("Received response!")
                    
                    guard let httpResponse = response as? HTTPURLResponse else {
                        print("Response is not HTTP")
                        throw GeminiError.invalidResponse
                    }
                    
                    print("HTTP Status: \(httpResponse.statusCode)")
                    
                    guard httpResponse.statusCode == 200 else {
                        print("Bad HTTP status: \(httpResponse.statusCode)")
                        throw GeminiError.invalidResponse
                    }
                    
                    print("Got HTTP 200, response size: \(data.count) bytes")
                    
                    // Parse the JSON response
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        print("Parsed as JSON array with \(json.count) items")
                        
                        var buffer = ""
                        
                        // Extract text from all response chunks
                        for item in json {
                            if let candidates = item["candidates"] as? [[String: Any]],
                               let content = candidates.first?["content"] as? [String: Any],
                               let parts = content["parts"] as? [[String: Any]],
                               let text = parts.first?["text"] as? String {
                                
                                buffer += text
                                print("Added to buffer: \(text)")
                            }
                        }
                        
                        print("Final buffer: \(buffer)")
                        
                        // Extract all thoughts from the complete buffer
                        while let thoughtRange = extractThought(from: buffer) {
                            let thought = String(buffer[thoughtRange.thought])
                            print("Extracted thought: \(thought)")
                            continuation.yield(thought)
                            buffer.removeSubrange(thoughtRange.full)
                        }
                    }
                    
                    print("Finishing continuation...")
                    continuation.finish()
                    
                } catch {
                    print("Error in streamThoughts: \(error)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    private func buildPrompt(transcript: String) -> String {
        // Load template if not already cached
        if promptTemplate == nil {
            do {
                promptTemplate = try loadTemplate()
            } catch {
                print("Could not load template: \(error)")
                // Fallback to basic template if file loading fails
                return "Based on this conversation, provide thoughts in [bt]thought[et] format:\n\(transcript)"
            }
        }
        
        guard let template = promptTemplate else {
            return "Based on this conversation, provide thoughts in [bt]thought[et] format:\n\(transcript)"
        }
        
        // Replace {{conversation}} with actual transcript
        return template.replacingOccurrences(of: "{{conversation}}", with: transcript)
    }

    // Add back the loadTemplate method
    private func loadTemplate() throws -> String {
        guard let templateURL = Bundle.main.url(forResource: "gemini_prompt_template", withExtension: "txt") else {
            throw GeminiError.templateNotFound
        }
        
        return try String(contentsOf: templateURL, encoding: .utf8)
    }
    
    private func buildRequest(prompt: String) -> URLRequest {
        let url = URL(string: baseURL + "?key=\(apiKey)")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        
        let body: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        ["text": prompt]
                    ]
                ]
            ]
        ]
        
        request.httpBody = try! JSONSerialization.data(withJSONObject: body)
        return request
    }
    
    private func extractThought(from buffer: String) -> (full: Range<String.Index>, thought: Range<String.Index>)? {
        guard let btStart = buffer.range(of: "[bt]"),
              let etEnd = buffer.range(of: "[et]", range: btStart.upperBound..<buffer.endIndex) else {
            return nil
        }
        
        let fullRange = btStart.lowerBound..<etEnd.upperBound
        let thoughtRange = btStart.upperBound..<etEnd.lowerBound
        return (full: fullRange, thought: thoughtRange)
    }
}

enum GeminiError: Error {
    case invalidURL
    case invalidResponse
    case noContent
    case templateNotFound
}
