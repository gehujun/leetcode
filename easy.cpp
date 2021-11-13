#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <cmath>

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

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:

    template<typename T>
    static void output(T values){ cout<<values<<"\t";}

    //94. 二叉树的中序遍历
    void recursion(TreeNode* node,vector<int>& vec){
        if(node->left != nullptr) recursion(node->left,vec);
        vec.push_back(node->val);
        if(node->right != nullptr) recursion(node->right,vec);
    }

    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        if(root== nullptr) return {};
        recursion(root,ans);
        return ans;
    }

    //338. 比特位计数
    vector<int> countBits(int n) {
        if(n==0) return {0};
        if(n==1) return {0,1};
        vector<int> ans{0,1};
        int m = 2;
        for(int i= 2,j=0;i<=n;i++,j++){
            if(i == m){
                m = m*2;
                j=0;
            }
            ans.push_back(ans[j]+1);
        }
        return ans;
    }

    //121. 买卖股票的最佳时机
    int maxProfit(vector<int>& prices,int type) {
        if(type==1){
            //121. 买卖股票的最佳时机
            int ans{0},index{0};
            for(int i=0;i<prices.size();i++){
                int tmp = prices[i]-prices[index];
                if(tmp>ans) ans = tmp;
                else if(tmp<=0) index = i;
            }
            return ans;
        }else if(type==2){
            //122. 买卖股票的最佳时机 II
            int ans{0};
            for(int i=1;i<prices.size();i++){
                int pro = prices[i]-prices[i-1];
                if(pro>0){
                    ans += pro;
                    continue;
                }
            }
            return ans;
        }else if(type==3){
            //309. 最佳买卖股票时机含冷冻期-动态规划
            //动态规划，每天的状态分为三种，1.持续持有；2.处于冷冻期；3.不处于冷冻期；
            if (prices.empty()) {
                return 0;
            }

            int n = prices.size();
            // f[i][0]: 手上持有股票的最大收益
            // f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
            // f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
            vector<vector<int>> f(n, vector<int>(3));
            f[0][0] = -prices[0];
            for (int i = 1; i < n; ++i) {
                f[i][0] = max(f[i - 1][0], f[i - 1][2] - prices[i]);
                f[i][1] = f[i - 1][0] + prices[i];
                f[i][2] = max(f[i - 1][1], f[i - 1][2]);
            }
            return max(f[n - 1][1], f[n - 1][2]);
                

        }
            return 0;
    }

    //21.合并两个有序链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == nullptr) return l2;
        if(l2 == nullptr) return l1;
        ListNode* head = new ListNode();
        
        ListNode* ans = head;
        for (;;){
            if(l1->val > l2->val){
                ans->next = l2;
                l2 = l2->next;
                ans = ans->next;
            }else{
                ans->next = l1;
                l1 = l1->next;
                ans = ans->next;
            }
            if(l1 == nullptr){
                ans->next = l2;
                break;
            }
            if(l2 == nullptr){
                ans->next = l1;
                break;
            }
        }
        ans = head->next;
        delete head;
        return ans;
    }
    
};

int main(){
    Solution s;
    
    //94. 二叉树的中序遍历
    // TreeNode* node1 = new TreeNode(1);
    // TreeNode* node2 = new TreeNode(2);
    // TreeNode* node3 = new TreeNode(3);
    // node1->right    = node2;
    // node2->left     = node3;
    // vector<int> ans;
    // ans = s.inorderTraversal(node1);
    // for(auto i : ans) {cout<<i<<"\t";}

    //338. 比特位计数
    // vector<int> ans;
    // ans = s.countBits(2);
    // for(auto i : ans) {cout<<i<<"\t";}

    //121. 买卖股票的最佳时机
    vector<int> vec{1,2,3,4,5};
    cout<<s.maxProfit(vec,3)<<endl;

}