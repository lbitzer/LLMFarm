//
//  ChatViewModel.swift
//
//  Created by Artem Savkin
//

import Foundation
import SwiftUI
import SimilaritySearchKit
import SimilaritySearchKitDistilbert
import SimilaritySearchKitMiniLMAll
import SimilaritySearchKitMiniLMMultiQA
import os
import llmfarm_core

private extension Duration {
    var seconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1.0e18
    }
}

var AIChatModel_obj_ptr: UnsafeMutableRawPointer? = nil

@MainActor
final class AIChatModel: ObservableObject {
    
    enum State {
        case none
        case loading
        case ragIndexLoading
        case ragSearch
        case completed
    }
    
    public var chat: AI?
    public var modelURL: String = ""
    public var numberOfTokens = 0
    public var total_sec = 0.0
    public var action_button_icon = "paperplane"
    public var model_loading = false
    public var model_name = ""
    public var chat_name = ""
    public var start_predicting_time = DispatchTime.now()
    public var first_predicted_token_time = DispatchTime.now()
    public var tok_sec: Double = 0.0
    public var ragIndexLoaded: Bool = false
    private var state_dump_path: String = ""
    private var title_backup = ""
    private var messages_lock = NSLock()
    public var ragUrl: URL
    private var ragTop: Int = 3
    private var chunkSize: Int = 256
    private var chunkOverlap: Int = 100
    private var currentModel: EmbeddingModelType = .minilmMultiQA
    private var comparisonAlgorithm: SimilarityMetricType = .dotproduct
    private var chunkMethod: TextSplitterType = .recursive
    
    @Published var predicting = false
    @Published var AI_typing = 0
    @Published var state: State = .none
    @Published var messages: [Message] = []
    @Published var load_progress: Float = 0.0
    @Published var Title: String = ""
    @Published var is_mmodal: Bool = false
    @Published var cur_t_name: String = ""
    @Published var cur_eval_token_num: Int = 0
    @Published var query_tokens_count: Int = 0
    
    @Published var geminiThinking = false
    private let geminiService = GeminiService()
    private var conversationTranscript: [String] = []
    
    public init() {
        let ragDir = GetRagDirRelPath(chat_name: self.chat_name)
        ragUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(ragDir) ?? URL(fileURLWithPath: "")
    }
    
    public func ResetRAGUrl() {
        let ragDir = GetRagDirRelPath(chat_name: self.chat_name)
        ragUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(ragDir) ?? URL(fileURLWithPath: "")
    }
    
    private func model_load_progress_callback(_ progress: Float) -> Bool {
        DispatchQueue.main.async {
            self.load_progress = progress
        }
        return true
    }
    
    private func eval_callback(_ t: Int) -> Bool {
        DispatchQueue.main.async {
            if t == 0 {
                self.cur_eval_token_num += 1
            }
        }
        return false
    }
    
    private func after_model_load(_ load_result: String, in_text: String, attachment: String? = nil, attachment_type: String? = nil) {
        guard load_result == "[Done]", let chatModel = self.chat?.model, chatModel.context != nil else {
            self.finish_load(append_err_msg: true, msg_text: "Load Model Error: \(load_result)")
            return
        }
        
        self.finish_load()
        var system_prompt: String? = nil
        if chatModel.contextParams.system_prompt != "", chatModel.nPast == 0 {
            system_prompt = chatModel.contextParams.system_prompt  + "\n"
            self.messages[self.messages.endIndex - 1].header = chatModel.contextParams.system_prompt
        }
        chatModel.parse_skip_tokens()
        Task {
            await self.Send(message: in_text, append_user_message: false, system_prompt: system_prompt, attachment: attachment, attachment_type: attachment_type)
        }
    }
    
    public func hard_reload_chat() {
        self.remove_dump_state()
        self.chat?.model?.contextParams.save_load_state = false
        self.chat = nil
    }
    
    public func remove_dump_state() {
        if FileManager.default.fileExists(atPath: self.state_dump_path) {
            try? FileManager.default.removeItem(atPath: self.state_dump_path)
        }
    }
    
    public func reload_chat(_ chat_selection: Dictionary<String, String>) {
        self.stop_predict()
        self.chat_name = chat_selection["chat"] ?? "Not selected"
        self.Title = chat_selection["title"] ?? ""
        self.is_mmodal = chat_selection["mmodal"] == "1"
        messages_lock.lock()
        self.messages = load_chat_history(chat_selection["chat"]! + ".json") ?? []
        messages_lock.unlock()
        self.state_dump_path = get_state_path_by_chat_name(chat_name) ?? ""
        ResetRAGUrl()
        self.ragIndexLoaded = false
        self.AI_typing = -Int.random(in: 0..<100000)
    }
    
    public func update_chat_params() {
        guard let chat_config = getChatInfo(self.chat?.chatName ?? "") else { return }
        self.chat?.model?.contextParams = get_model_context_param_by_config(chat_config)
        self.chat?.model?.sampleParams = get_model_sample_param_by_config(chat_config)
    }
    
    public func load_model_by_chat_name_prepare(_ chat_name: String, in_text: String, attachment: String? = nil, attachment_type: String? = nil) -> Bool? {
        guard let chat_config = getChatInfo(chat_name), let model_inference = chat_config["model_inference"], let model = chat_config["model"] else {
            return nil
        }
        
        self.model_name = model as! String
        guard let m_url = get_path_by_short_name(self.model_name) else {
            return nil
        }
        self.modelURL = m_url
        
        var model_sample_param = get_model_sample_param_by_config(chat_config)
        var model_context_param = get_model_context_param_by_config(chat_config)
        
        if let grammar = chat_config["grammar"] as? String, grammar != "<None>", grammar != "" {
            model_context_param.grammar_path = get_grammar_path_by_name(grammar)
        }
        
        self.chunkSize = chat_config["chunk_size"] as? Int ?? self.chunkSize
        self.chunkOverlap = chat_config["chunk_overlap"] as? Int ?? self.chunkOverlap
        self.ragTop = chat_config["rag_top"] as? Int ?? self.ragTop
        self.currentModel = getCurrentModelFromStr(chat_config["current_model"] as? String ?? "")
        self.comparisonAlgorithm = getComparisonAlgorithmFromStr(chat_config["comparison_algorithm"] as? String ?? "")
        self.chunkMethod = getChunkMethodFromStr(chat_config["chunk_method"] as? String ?? "")
        
        AIChatModel_obj_ptr = nil
        self.chat = AI(_modelPath: modelURL, _chatName: chat_name)
        guard let chat = self.chat else {
            return nil
        }
        chat.initModel(model_context_param.model_inference, contextParams: model_context_param)
        guard let chatModel = chat.model else {
            return nil
        }
        chatModel.sampleParams = model_sample_param
        chatModel.contextParams = model_context_param
        
        return true
    }
    
    public func load_model_by_chat_name(_ chat_name: String, in_text: String, attachment: String? = nil, attachment_type: String? = nil) -> Bool? {
        self.model_loading = true
        
        if self.chat?.model?.contextParams.save_load_state == true {
            self.chat?.model?.contextParams.state_dump_path = get_state_path_by_chat_name(chat_name) ?? ""
        }
        
        self.chat?.model?.modelLoadProgressCallback = { progress in
            return self.model_load_progress_callback(progress)
        }
        self.chat?.model?.modelLoadCompleteCallback = { load_result in
            self.chat?.model?.evalCallback = self.eval_callback
            self.after_model_load(load_result, in_text: in_text, attachment: attachment, attachment_type: attachment_type)
        }
        self.chat?.loadModel()
        
        return true
    }
    
    private func update_last_message(_ message: inout Message) {
        messages_lock.lock()
        if let _ = messages.last {
            messages[messages.endIndex - 1] = message
        }
        messages_lock.unlock()
    }
    
    public func save_chat_history_and_state() {
        save_chat_history(self.messages, self.chat_name + ".json")
        self.chat?.model?.save_state()
    }
    
    public func stop_predict(is_error: Bool = false) {
        self.chat?.flagExit = true
        self.total_sec = Double((DispatchTime.now().uptimeNanoseconds - self.start_predicting_time.uptimeNanoseconds)) / 1_000_000_000
        if let last_message = messages.last {
            messages_lock.lock()
            if last_message.state == .predicting || last_message.state == .none {
                messages[messages.endIndex - 1].state = .predicted(totalSecond: self.total_sec)
                messages[messages.endIndex - 1].tok_sec = Double(self.numberOfTokens) / self.total_sec
            }
            if is_error {
                messages[messages.endIndex - 1].state = .error
            }
            messages_lock.unlock()
        }
        self.predicting = false
        self.tok_sec = Double(self.numberOfTokens) / self.total_sec
        self.numberOfTokens = 0
        self.action_button_icon = "paperplane"
        self.AI_typing = 0
        self.save_chat_history_and_state()
        if is_error {
            self.chat = nil
        }
    }
    
    public func check_stop_words(_ token: String, _ message_text: inout String) -> Bool {
        for stop_word in self.chat?.model?.contextParams.reverse_prompt ?? [] {
            if token == stop_word || message_text.hasSuffix(stop_word) {
                if stop_word.count > 0 && message_text.count > stop_word.count {
                    message_text.removeLast(stop_word.count)
                }
                return false
            }
        }
        return true
    }
    
    public func process_predicted_str(_ str: String, _ time: Double, _ message: inout Message) -> Bool {
        let check = check_stop_words(str, &message.text)
        if !check {
            self.stop_predict()
        }
        if check, self.chat?.flagExit != true, self.chat_name == self.chat?.chatName {
            message.state = .predicting
            message.text += str
            self.AI_typing += 1
            update_last_message(&message)
            self.numberOfTokens += 1
        } else {
            print("chat ended.")
        }
        return check
    }
    
    public func finish_load(append_err_msg: Bool = false, msg_text: String = "") {
        if append_err_msg {
            self.messages.append(Message(sender: .system, state: .error, text: msg_text, tok_sec: 0))
            self.stop_predict(is_error: true)
        }
        self.state = .completed
        self.Title = self.title_backup
    }
    
    public func finish_completion(_ final_str: String, _ message: inout Message) {
        self.cur_t_name = ""
        self.load_progress = 0
        print(final_str)
        self.AI_typing = 0
        self.total_sec = Double((DispatchTime.now().uptimeNanoseconds - self.start_predicting_time.uptimeNanoseconds)) / 1_000_000_000
        if self.chat_name == self.chat?.chatName, self.chat?.flagExit != true {
            message.tok_sec = self.tok_sec != 0 ? self.tok_sec : Double(self.numberOfTokens) / self.total_sec
            message.state = .predicted(totalSecond: self.total_sec)
            update_last_message(&message)
        } else {
            print("chat ended.")
        }
        self.predicting = false
        self.numberOfTokens = 0
        self.action_button_icon = "paperplane"
        if final_str.hasPrefix("[Error]") {
            self.messages.append(Message(sender: .system, state: .error, text: "Eval \(final_str)", tok_sec: 0))
        }
        self.save_chat_history_and_state()
    }
    
    public func LoadRAGIndex(ragURL: URL) async {
        updateIndexComponents(currentModel: currentModel, comparisonAlgorithm: comparisonAlgorithm, chunkMethod: chunkMethod)
        await loadExistingIndex(url: ragURL, name: "RAG_index")
        ragIndexLoaded = true
    }
    
    public func RegenerateLstMessage() {
        // self.messages.removeLast()
    }
    
    public func GenerateRagLLMQuery(_ inputText: String, _ searchResultsCount: Int, _ ragURL: URL, message in_text: String, append_user_message: Bool = true, system_prompt: String? = nil, attachment: String? = nil, attachment_type: String? = nil) {
        let aiQueue = DispatchQueue(label: "LLMFarm-RAG", qos: .userInitiated, attributes: .concurrent)
        
        aiQueue.async {
            Task {
                if await !self.ragIndexLoaded {
                    await self.LoadRAGIndex(ragURL: ragURL)
                }
                DispatchQueue.main.async {
                    self.state = .ragSearch
                }
                let results = await searchIndexWithQuery(query: inputText, top: searchResultsCount)
                let llmPrompt = SimilarityIndex.exportLLMPrompt(query: inputText, results: results!)
                await self.Send(message: llmPrompt, append_user_message: false, system_prompt: system_prompt, attachment: llmPrompt, attachment_type: "rag")
            }
        }
    }
    
    public func SetSendMsgTokensCount(_ count: Int) {
        // Implementation here
    }
    
    public func SetGeneratedMsgTokensCount(_ count: Int) {
        // Implementation here
    }
    
    public func Send(message in_text: String, append_user_message: Bool = true, system_prompt: String? = nil, attachment: String? = nil, attachment_type: String? = nil, useRag: Bool = false) async {
        
        await processWithGeminiAndSmolLM(in_text)
        
        self.AI_typing += 1
        
        if append_user_message {
            let requestMessage = Message(sender: .user, state: .typed, text: in_text, tok_sec: 0, attachment: attachment, attachment_type: attachment_type)
            self.messages.append(requestMessage)
        }
        
        if self.chat == nil {
            guard let _ = load_model_by_chat_name_prepare(chat_name, in_text: in_text, attachment: attachment, attachment_type: attachment_type) else {
                return
            }
        }
        
        if useRag {
            self.state = .ragIndexLoading
            self.GenerateRagLLMQuery(in_text, self.ragTop, self.ragUrl, message: in_text, append_user_message: append_user_message, system_prompt: system_prompt, attachment: attachment, attachment_type: attachment_type)
            return
        }
        
        if self.chat?.model?.context == nil {
            self.state = .loading
            title_backup = Title
            Title = "loading..."
            let res = self.load_model_by_chat_name(self.chat_name, in_text: in_text, attachment: attachment, attachment_type: attachment_type)
            if res == nil {
                finish_load(append_err_msg: true, msg_text: "Model load error")
            }
            return
        }
        
        if attachment != nil, attachment_type == "rag" {
            let requestMessage = Message(sender: .user_rag, state: .typed, text: in_text, tok_sec: 0, attachment: attachment, attachment_type: attachment_type)
            self.messages.append(requestMessage)
        }
        
        self.state = .completed
        self.chat?.chatName = self.chat_name
        self.chat?.flagExit = false
        var message = Message(sender: .system, text: "", tok_sec: 0)
        self.messages.append(message)
        self.numberOfTokens = 0
        self.total_sec = 0.0
        self.predicting = true
        self.action_button_icon = "stop.circle"
        let img_real_path = get_path_by_short_name(attachment ?? "unknown", dest: "cache/images")
        self.start_predicting_time = DispatchTime.now()
        self.chat?.conversation(in_text, { str, time in
            _ = self.process_predicted_str(str, time, &message)
        }, { key, value in
            // Handle key-value pairs if needed
        }, { final_str in
            self.finish_completion(final_str, &message)
        }, system_prompt: system_prompt, img_path: img_real_path)
    }
    
    
    // Build ChatML prompt for SmolLM (same format as your Python script)
    private func buildSmolLMPrompt(userInput: String, thoughts: [String], responses: [String]) -> String {
        var prompt = "<|im_start|>user\n\(userInput)<|im_end|>\n"
        
        if responses.isEmpty {
            // First thought, no previous responses
            if !thoughts.isEmpty {
                prompt += "<|im_start|>knowledge\n\(thoughts[0])<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"  // Add this line to prompt for assistant response
            }
        } else {
            // Multi-turn: build history of thought->response pairs
            let numResponses = responses.count
            for i in 0..<numResponses {
                prompt += "<|im_start|>knowledge\n\(thoughts[i])<|im_end|>\n"
                prompt += "<|im_start|>assistant\n\(responses[i])<|im_end|>\n"
            }
            // Add current thought if we have more thoughts than responses
            if thoughts.count > responses.count {
                prompt += "<|im_start|>knowledge\n\(thoughts[responses.count])<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"  // Add this line
            }
        }
        
        return prompt
    }
    
    private func generateLocalResponse(prompt: String) async -> String? {
        return await withCheckedContinuation { (continuation: CheckedContinuation<String?, Never>) in
            guard let chat = self.chat else {
                continuation.resume(returning: nil)
                return
            }
            
            var result = ""
            chat.conversation(prompt, { str, time in
                result += str
                // Don't return anything - the closure expects Void
            }, { key, value in
                // Handle key-value pairs if needed
            }, { final_str in
                continuation.resume(returning: result)
            }, system_prompt: nil)  // Add the missing system_prompt parameter
        }
    }
    
    // Clean SmolLM response (remove ChatML tokens)
    private func cleanSmolLMResponse(_ response: String) -> String {
        var cleaned = response
        
        // Remove all variations of assistant tokens
        cleaned = cleaned.replacingOccurrences(of: "<|im_start|>assistant\n", with: "")
        cleaned = cleaned.replacingOccurrences(of: "<|im_start|>assistant", with: "")
        cleaned = cleaned.replacingOccurrences(of: "<|im_end|>", with: "")
        cleaned = cleaned.replacingOccurrences(of: "assistant\n", with: "")
        cleaned = cleaned.replacingOccurrences(of: "assistant", with: "")
        
        // Clean up whitespace and newlines
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Remove multiple spaces
        cleaned = cleaned.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
        
        return cleaned
    }
    
    // Shared state actor
    actor SharedState {
        private var thoughts: [String] = []
        private var responses: [String] = []
        private var fullResponse = ""
        private var isGeminiDone = false
        
        func addThought(_ thought: String) {
            thoughts.append(thought)
        }
        
        func addResponse(_ response: String) {
            responses.append(response)
            fullResponse += (fullResponse.isEmpty ? "" : " ") + response
        }
        
        func getThoughts() -> [String] { thoughts }
        func getResponses() -> [String] { responses }
        func getFullResponse() -> String { fullResponse }
        
        func setGeminiDone() { isGeminiDone = true }
        func isGeminiComplete() -> Bool { isGeminiDone }
        func getProcessedCount() -> Int { responses.count }
        func getAvailableThoughts() -> Int { thoughts.count }
    }

    private func processWithGeminiAndSmolLM(_ input: String) async {
        let geminiService = GeminiService()
        
        // FIRST: Ensure model is loaded before starting dual-model processing
        if self.chat?.model?.context == nil {
            print("DEBUG: Model not loaded, loading now...")
            
            // Set the correct model path
            self.modelURL = "/Users/lawrencebitzer/Documents/Conversational/conv-fill-06202025.gguf"
            self.model_name = "conv-fill-06202025.gguf"
            
            // Load the model synchronously first
            if let _ = load_model_by_chat_name_prepare(self.chat_name, in_text: input) {
                print("DEBUG: Model prepare completed, now loading...")
                let _ = load_model_by_chat_name(self.chat_name, in_text: input)
                
                // Wait for model to finish loading
                while self.model_loading {
                    print("DEBUG: Waiting for model to load...")
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
                }
                
                if self.chat?.model?.context != nil {
                    print("DEBUG: Model loaded successfully!")
                } else {
                    print("DEBUG: Model failed to load")
                    return
                }
            } else {
                print("DEBUG: Model prepare failed")
                return
            }
        } else {
            print("DEBUG: Model already loaded")
        }
        
        // Create initial response message
        var responseMessage = Message(sender: .system, text: "Thinking...", tok_sec: 0)
        self.messages.append(responseMessage)
        
        // Shared state between threads
        let sharedState = SharedState()
        
        // Build conversation context
        let transcript = conversationTranscript.joined(separator: "\n") +
                        (conversationTranscript.isEmpty ? "" : "\n") +
                        "User: \(input)"
        
        // Start Gemini task
        async let geminiTask: () = Task {
            do {
                for try await thought in geminiService.streamThoughts(transcript: transcript) {
                    if thought == "[done]" {
                        print("DEBUG: Gemini received [done] signal")
                        await sharedState.setGeminiDone()
                        break
                    }
                    print("Gemini produced thought: \(thought)")
                    await sharedState.addThought(thought)
                }
                print("DEBUG: Gemini stream completed, setting done flag")
                await sharedState.setGeminiDone()
            } catch {
                print("Gemini error: \(error)")
                await sharedState.setGeminiDone()
            }
        }.value
        
        // Start SmolLM task
        async let smolLMTask: () = Task {
            await processSmolLMThoughts(input: input, sharedState: sharedState, chatModel: self)
        }.value
        
        // Wait for both tasks to complete
        _ = await (geminiTask, smolLMTask)
        
        // Final UI update - mark as completed
        let finalResponse = await sharedState.getFullResponse()
        await MainActor.run {
            responseMessage.state = .predicted(totalSecond: 0.0)
            responseMessage.tok_sec = 0.0
            responseMessage.text = finalResponse
            self.messages[self.messages.count - 1] = responseMessage
        }
        
        // Add to conversation transcript
        if !finalResponse.isEmpty {
            conversationTranscript.append("User: \(input)")
            conversationTranscript.append("Assistant: \(finalResponse)")
        }
        
        print("Dual-model processing completed")
    }

    private func processSmolLMThoughts(input: String, sharedState: SharedState, chatModel: AIChatModel) async {
        print("DEBUG: SMOLLM TASK STARTED")
        var loopCount = 0
        let maxLoops = 100
        
        while loopCount < maxLoops {
            print("DEBUG: STARTING LOOP ITERATION \(loopCount)")
            loopCount += 1
            
            let thoughtCount = await sharedState.getAvailableThoughts()
            let processedCount = await sharedState.getProcessedCount()
            let isGeminiDone = await sharedState.isGeminiComplete()
            
            print("DEBUG: Loop \(loopCount) - thoughtCount=\(thoughtCount), processedCount=\(processedCount), isGeminiDone=\(isGeminiDone)")
            
            if isGeminiDone && processedCount >= thoughtCount {
                print("SmolLM finished - no more thoughts to process")
                break
            }
            
            if processedCount < thoughtCount {
                let thoughts = await sharedState.getThoughts()
                let responses = await sharedState.getResponses()
                let currentThought = thoughts[processedCount]
                
                print("DEBUG: Processing thought \(processedCount): '\(currentThought)'")
                
                // Build prompt
                var prompt = "<|im_start|>user\n\(input)<|im_end|>\n"
                let thoughtsToUse = Array(thoughts.prefix(processedCount + 1))
                
                if responses.isEmpty {
                    if !thoughtsToUse.isEmpty {
                        prompt += "<|im_start|>knowledge\n\(thoughtsToUse[0])<|im_end|>\n"
                        prompt += "<|im_start|>assistant\n"
                    }
                } else {
                    let numResponses = responses.count
                    for i in 0..<numResponses {
                        prompt += "<|im_start|>knowledge\n\(thoughtsToUse[i])<|im_end|>\n"
                        prompt += "<|im_start|>assistant\n\(responses[i])<|im_end|>\n"
                    }
                    if thoughtsToUse.count > responses.count {
                        prompt += "<|im_start|>knowledge\n\(thoughtsToUse[responses.count])<|im_end|>\n"
                        prompt += "<|im_start|>assistant\n"
                    }
                }
                
                print("DEBUG: Built prompt, calling generateLocalResponse...")
                
                // Generate response using local model
                if let response = await chatModel.generateLocalResponse(prompt: prompt) {
                    // Clean the response
                    var cleanResponse = response
                    cleanResponse = cleanResponse.replacingOccurrences(of: "<|im_start|>assistant\n", with: "")
                    cleanResponse = cleanResponse.replacingOccurrences(of: "<|im_start|>assistant", with: "")
                    cleanResponse = cleanResponse.replacingOccurrences(of: "<|im_end|>", with: "")
                    cleanResponse = cleanResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    if !cleanResponse.isEmpty {
                        print("DEBUG: Generated response: '\(cleanResponse)'")
                        await sharedState.addResponse(cleanResponse)
                        
                        // Update UI immediately
                        let fullResponse = await sharedState.getFullResponse()
                        await MainActor.run {
                            var currentMessage = chatModel.messages[chatModel.messages.count - 1]
                            currentMessage.text = fullResponse
                            currentMessage.state = .predicting
                            chatModel.messages[chatModel.messages.count - 1] = currentMessage
                        }
                    } else {
                        print("DEBUG: Empty response, using fallback")
                        await sharedState.addResponse("I understand.")
                    }
                } else {
                    print("DEBUG: generateLocalResponse returned nil, using fallback")
                    await sharedState.addResponse("That's interesting!")
                }
                
                print("DEBUG: Added response successfully")
            } else {
                print("DEBUG: Waiting for more thoughts...")
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        
        print("DEBUG: SMOLLM TASK FINISHED")
    }
}
