import torch


class CharacterTokenizer:

    def __init__(self):
        self.chars = ['[PAD]', '[UKN]', '[ST]', '[ED]', '\n', ' ', 'V', 'a', 'l', 'k', 'y', 'r', 'i', 'C', 'h', 'o',
                      'n', 'c', 'e', 's', 'I', 'S', 'j', '3', '(', 'J', 'p', ',', 't', '.', 'f', 'B', 'd', ')', 'm',
                      'u', 'g', 'v', 'b', 'M', 'P', 'R', '2', '0', '1', 'E', 'w', 'N', 'T', 'W', 'H', 'O', 'z', 'A',
                      'G', 'x', 'D', 'U', 'L', 'q', '4', 'F', 'K', '9', '8', '7', '6', '5', 'Q', 'Y', 'X', 'Z']
        self.char_map = {}
        for char in self.chars:
            self.char_map[char] = self.chars.index(char)
        self.pad_token_id = 0
        self.unk_token = 1
        self.start_token = 2
        self.end_token = 3

    def add_char(self, char):
        if char in self.char_map:
            return
        self.chars.append(char)
        self.char_map[char] = len(self.chars) - 1

    def encode(self, list_of_strings, max_length):
        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), self.pad_token_id, dtype=torch.long)
        for idx, string in enumerate(list_of_strings):
            input_ids[idx, 0] = self.start_token
            attention_masks[idx, 0] = 1

            for char_idx, char in enumerate(string):
                if char in self.char_map:
                    input_ids[idx, char_idx + 1] = self.char_map[char]
                else:
                    input_ids[idx, char_idx + 1] = self.unk_token
                attention_masks[idx, char_idx + 1] = 1

            input_ids[idx, len(string) + 1] = self.end_token
            attention_masks[idx, len(string) + 1] = 1

        return input_ids, attention_masks

    def decode(self, outputs_ids):
        return ''.join(self.chars[x] for x in outputs_ids)
