#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <limits.h>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

ListNode* creatList(vector<int> vec){
    if(vec.size()==0) return nullptr;
    ListNode* head= new ListNode(vec[0]);
    ListNode* curr = head;
    for(int i = 1;i<vec.size();i++){
        curr->next = new ListNode(vec[i]);
        curr = curr->next;
    }
    return head;
}

void printList(ListNode* head){
    ListNode* curr = head;
    while(curr!=nullptr){
        cout<<curr->val<<"\t";
        curr = curr->next;
    }
    cout<<endl;
}

class Solution {
public:

    template<typename T>
    static void output(T values){ cout<<values<<"\t";}

    int findUnsortedSubarray(vector<int>& nums) {
        if(nums.size()<2) return 0;
        int l=nums.size(),r=-1;
        for(int i=0;i<nums.size()-1;i++){
            if((nums[i]>nums[i+1])) {
                if(i<l) l = i;
                if(i>=r) r = i+1;
            }
        }
        if(l>r) return 0;
        int min=nums[l],max=nums[l];
        for(int i = l;i<=r;i++){
            if(nums[i]>max) max = nums[i];
            if(nums[i]<min) min = nums[i];
        }
        while(l>0){
            if(nums[l-1]>min) l--;
            else break;
        }
        while(r<(nums.size()-1)){
            if(nums[r+1]<max) r++;
            else break;
        }
        // cout<<l<<"\t"<<r<<endl;
        return (r-l+1);
    }

    //逆拓扑排序的题
    vector<int> eventualSafeNodes(vector<vector<int>>& graph);

    //9宫格打字-我的解答，占用空间较多 可以用回溯法解题使用到的内存较小
    vector<string> letterCombinations(string digits) {
        if(digits.size() == 0) return {};
        //电话号码字典
        vector<vector<char>> dic = {
            {},
            {},
            {'a','b','c'},
            {'d','e','f'},
            {'g','h','i'},
            {'j','k','l'},
            {'m','n','o'},
            {'p','q','r','s'},
            {'t','u','v'},
            {'w','x','y','z'}
        };
        //处理输入
        vector<int> inputs;
        for(char c : digits){
            inputs.push_back(c-'0');
        }
        //一个个的处理输入的号码
        vector<vector<string>> records(inputs.size(),vector<string>{});
        for_each(dic[inputs[0]].begin(),dic[inputs[0]].end(),
            [&](char c){string s(1,c);records[0].push_back(s);} );
        for(int i=1;i<inputs.size();i++){
            for(char c : dic[inputs[i]]){
                for(string str : records[i-1]){
                    records[i].push_back(str+=c);
                }
            }
        }
        
        return records[records.size()-1];
    }

    //删除链表的倒数第 N 个结点
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        
        ListNode* root = new ListNode(0,head);
        int len = getLength(head);

        ListNode* cnt = root;
        for(int i = 0;i<len-n;i++){
            cnt = cnt->next;
        }

        cnt->next = cnt->next->next;
        ListNode* ans = root->next;
        delete root;
        return ans;
    }
    int getLength(ListNode* head) {
        int length = 0;
        while (head) {
            ++length;
            head = head->next;
        }
        return length;
    }

    // ListNode* removeNthFromEnd(ListNode* head, int n) {
    //     ListNode* dummy = new ListNode(0, head);
    //     int length = getLength(head);
    //     ListNode* cur = dummy;
    //     for (int i = 1; i < length - n + 1; ++i) {
    //         cur = cur->next;
    //     }
    //     cur->next = cur->next->next;
    //     ListNode* ans = dummy->next;
    //     delete dummy;
    //     return ans;
    // }

    //括号生成
    // bool valid(string str){
    //     int balance = 0;
    //     for(char c :str){
    //         if(c == '('){
    //             balance += 1;
    //         } else {
    //             balance -= -1;
    //         }
    //         if(balance <0) return false;
    //     }
    //     return true;
    // }
    // vector<string> generateParenthesis(int n) {
        
    // }

    //搜索旋转排序数组-暴力法
    // int search(vector<int>& nums, int target) {
    //     for(int i = 0; i<nums.size();i++){
    //         if(nums[i] == target){
    //             return i;
    //         }
    //     }
    //     return -1;
    // }
    // //二分递归法
    int search(vector<int>& nums, int target){
        int l = 0,r = nums.size()-1;
        int n = nums.size();
        if(nums.size() ==0) return -1;
        if(nums.size() ==1) {
            (nums[0]==target) ? 0 : -1;
        }
        while(l<=r){
            int mid = (r-l)/2;
            if(nums[mid]==target) return mid;
            if(nums[0]<nums[mid]){
                if(nums[0]<=target && nums[mid]>=target){
                    r = mid -1;
                }else{
                    l = mid+1;
                }
            }else{
                if(target>nums[mid] && target<=nums[n-1]){
                    l = mid+1;
                }else{
                    r = mid-1;
                }
            }
        }
        return -1;
    }

    //在排序数组中查找元素的第一个和最后一个位置
    vector<int> searchRange(vector<int>& nums, int target) {
        int n = nums.size();
        if(n == 0) return {-1,-1};
        if(n == 1) {
            if(nums[0]==target) return {0,0};
            else return {-1,-1};
        }
        vector<int> ans = {-1,-1};
        int l = 0,r = n-1;
        int mid;
        while(l<=r){
            mid = (r - l)/2 + l;
            if(nums[mid]==target){
                ans[0] = mid;
                ans[1] = mid;
                break;
            }else{
                if(nums[mid]>target){
                    r = mid-1;
                }else{
                    l = mid+1;
                }
            }
        }
        if(ans[0] == -1) return {-1,-1};
        l = r = mid;
        while(nums[l]==target && --l>=0){}
        while (nums[r]==target && ++r<=n-1){}
        return {l+1,r-1};
    }

    //414. Third Maximum Number
    int thirdMax(vector<int>& nums) {
        
        int first{LONG_MIN},second{LONG_MIN},third{LONG_MIN};
        for(auto n : nums){
            if(n>third&&n<second){
                third = n;
            }else if(n>second && n<first){
                third = second;
                second = n;
            }else if(n>first){
                third = second;
                second = first;
                first = n;
            }
        }
        return third==0 ? first : third;
    }

    
    
};

int main(){
    Solution s;
    // cout<<s.isPalindrome("bb")<<endl;
    // cout<<s.longestPalindrome("aaaa");

    // vector<int> height{1,8,6,2,5,4,8,3,7};
    // cout<<s.maxArea(height);

    // vector<int> nums{0,0,0,0};
    // vector<vector<int>> res;
    // res = s.threeSum(nums);
    
    // vector<int> nums{1,3,5,4,2};
    // int ans;
    // ans = s.findUnsortedSubarray(nums);
    // cout<<ans<<endl;

    //9宫格打字
    // string digit = "2";
    // vector<string> ans;
    // ans = s.letterCombinations(digit);
    // for_each(ans.begin(),ans.end(),Solution::output<string>);

    //删除链表的倒数第 N 个结点
    // vector<int> vec{1};
    // ListNode* head = creatList(vec);
    // head = s.removeNthFromEnd(head,1);
    // printList(head);

    //搜索旋转数组
    // vector<int> vec{1};
    // int ans = s.search(vec,0);
    // cout<<ans;

    //在排序数组中查找元素的第一个和最后一个位置
    // vector<int> vec{2,2};
    // vector<int> ans = s.searchRange(vec,2);
    // cout<<ans[0]<<"\t"<<ans[1]<<endl;

    //414. Third Maximum Number
    vector<int> nums{5,2,2};
    int ans = s.thirdMax(nums);
    cout<<ans;

}