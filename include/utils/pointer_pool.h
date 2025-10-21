/**
Partly from Jacobus G.M. van der Linden “STreeD”
https://github.com/AlgTUDelft/pystreed
*/

#pragma once
#include "base.h"

namespace SORTD {

	template <class T>
	struct PointerPoolEntry {
		std::vector<T*> items;
		PointerPoolEntry* next;

		PointerPoolEntry(T* ptr) : next(nullptr) {
			items.reserve(1024);
			items.push_back(ptr);
		}
		~PointerPoolEntry() { Clear(); }

		inline bool HasSpace() const { return items.size() < items.capacity(); }
		
		void Add(T* ptr) {
			runtime_assert(HasSpace());
			items.push_back(ptr);
		}

		void Clear() {
			for (T* i: items) {
				delete i;
			}
			items.clear();
		}
	};
	
	template <class T>
	struct PointerPool {

		PointerPool() : head(nullptr), tail(nullptr) {}

		~PointerPool() {
			auto current = head;
			while (current) {
				auto* next = current->next;
				delete current;
				current = next;
			}
			head = nullptr;
			tail = nullptr;
		}

		// Disable copy constructor and assignment operator
		PointerPool(const PointerPool&) = delete;
		PointerPool& operator=(const PointerPool&) = delete;

		// Allow move semantics
		PointerPool(PointerPool&& other) noexcept
			: head(other.head), tail(other.tail) {
			other.head = nullptr;
			other.tail = nullptr;
		}

		PointerPool& operator=(PointerPool&& other) noexcept {
			if (this != &other) {
//				clear();
				head = other.head;
				tail = other.tail;
				other.head = nullptr;
				other.tail = nullptr;
			}
			return *this;
		}

		// Add a new pointer to the container
		void Add(T* ptr) {
			if (!head) {
				head = tail = new PointerPoolEntry<T>(ptr);
			} else if (tail->HasSpace()) {
				tail->Add(ptr);
			} else {
				auto entry = new PointerPoolEntry<T>(ptr);
				tail->next = entry;
				tail = entry;
			}
		}

		PointerPoolEntry<T>* head{ nullptr };
		PointerPoolEntry<T>* tail{ nullptr };
	};

}