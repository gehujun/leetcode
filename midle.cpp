#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <list>
#include <set>
#include <bitset>
#include <memory>
#include <queue>
#include <unordered_map>
#include <stack>
#include <limits>
using namespace std;


//Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


//以满二叉树的数值数组闯入，-1代表空值，构建二叉树。数值默认为正数
TreeNode* createTree(vector<int> vals){
    TreeNode* root;
    int n = vals.size();
    vector<TreeNode*> vec;
    vec.push_back(nullptr);
    for(int i = 0;i<n;i++){
        TreeNode* tmp = new TreeNode(vals[i]);
        vec.push_back(tmp);
    }

    for(int i=1;i<(n+1)/2;i++){
        int left = i*2;
        int right = i*2+1;
        if((left)<=n && vals[left] != -1){
            vec[i]->left = vec[left];
        }
        if((right)<=n && vals[right]!=-1){
            vec[i]->right = vec[right];
        }
    }
    return vec[1];
}

void printTree(TreeNode* root){
    //To do list:广度遍历
    queue<TreeNode*>que;
    que.emplace(root);
    while(!que.empty()){
        TreeNode* tmp = que.front();
        cout<<tmp->val<<" ";
        que.pop();
        if(tmp->left!= nullptr) que.emplace(tmp->left);
        if(tmp->right!= nullptr) que.emplace(tmp->right);
    }
    cout<<endl;

}

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

template<class T>
void printD2Vec(vector<vector<T>> ans){
    for(int i=0;i<ans.size();i++){
        for(int j=0;j<ans[i].size();j++){
            cout<<ans[i][j]<<"\t";
        }
        cout<<endl;
    }
}

class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        root  = createNode();
    }

    /** Inserts a word into the trie. */
    void insert(string word) {
        trieNode* curr = root;
        for(auto c : word){
            if(curr->childes[c-'a']==nullptr){
                trieNode* tmp = createNode();
                tmp->val = c;
                curr->childes[c-'a'] = tmp;
                curr = curr->childes[c-'a'];
            }else{
                curr = curr->childes[c-'a'];
            }
        }
        curr->isLeaf = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        trieNode* curr = root;
        for(auto c : word){
            curr = curr->childes[c-'a'];
            if(curr==nullptr) return false;
        }
        if(curr->isLeaf)    
            return true;
        return false;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        trieNode* curr = root;
        for(auto c : prefix){
            curr = curr->childes[c-'a'];
            if(curr==nullptr) return false;
        }
        return true;
    }

    void printTrie(){        
        queue<trieNode*> que;
        que.push(root);
        trieNode* curr;
        while(!que.empty()){
            curr = que.front();
            for(int i=0;i<26;i++){
                if(curr->childes[i]!=nullptr){
                    cout<<curr->childes[i]->val<<" ";
                    que.emplace(curr->childes[i]);
                }
            }   
            que.pop();
        }
        
    }

private:
    struct trieNode
    {
        bool isLeaf;
        char val;
        trieNode* childes[26];
        trieNode():isLeaf(false){};
        trieNode(char c):isLeaf(false),val(c){};        
    } ;
    struct trieNode* root;

    struct trieNode* createNode(){
        struct trieNode* tmp = new trieNode();
        for(int i=0;i<26;i++){
            tmp->childes[i] = NULL;
        }
        return tmp;
    }

};

// class trie{
// private:
//     bool isLeaf;
//     vector<trie*> childs;
// public:
//     trie(){

//     }


// };


class Solution {
public:

    //21.Merge Two Sorted List
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==nullptr) return l2;
        if(l2==nullptr) return l1;
        ListNode* dummyNode = new ListNode();
        ListNode* ptr1=l1,*ptr2=l2;
        ListNode* curr = dummyNode;
        for(;;){
            if(ptr1->val <= ptr2->val){
                curr->next = ptr1;
                curr = curr->next;
                ptr1 = ptr1->next;
                if(ptr1==nullptr){
                    curr->next = ptr2;
                    break;
                }
            }else{
                curr->next = ptr2;
                curr = curr->next;
                ptr2 = ptr2->next;
                if(ptr2 == nullptr){
                    curr->next = ptr1;
                    break;
                }
            }
        }
        return dummyNode->next;
    }

    int findUnsortedSubarray(vector<int>& nums) {
        if(nums.size()<2) return 0;
        int l=nums.size(),r=-1;
        for(int i=1;i<nums.size()-1;i++){
            if((nums[i] >= nums[i-1])&& (nums[i]<=nums[i+1])) continue;
            else {
                if(i<l) l = i;
                if(i>r) r = i;
            }
        }
        if((r-l) == (nums.size()-2)) return 0;
        else if((r-l)<0) return 0;
        else return (r-l+1);
    }

    //39. 组合总和
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        map<int,set<vector<int>>> res;
        res.insert( { 0 , {{}} } );
        sort(candidates.begin(),candidates.end());
        int len = candidates.size();
        for(int i = 1;i <= target;i++){
            set<vector<int>> tempAns;
            for(int j = 0;j<len;j++){
                int lastIndex = (i-candidates[j]);
                if(lastIndex<0) break;
                map<int,set<vector<int>>>::iterator lastAns;
                lastAns = res.find(lastIndex);
                if(lastAns!=res.end()){
                    // set<vector<int>> itr = lastAns->second;
                    for(auto itr : lastAns->second){
                        itr.push_back(candidates[j]);
                        sort(itr.begin(),itr.end());
                        tempAns.insert(itr);
                    }
                }
            }
            res.insert({i,tempAns});
        }
        vector<vector<int>> ans;
        set<vector<int>>::iterator itr=res[target].begin();
        for(;itr!=res[target].end();itr++){
            ans.push_back(*itr);
        }
        return ans;
    }

    //46. 全排列
    void backtrack(vector<int> nums,int index,vector<vector<int>>& res,int len){
        
        if(index == len-1) {
            res.emplace_back(nums);
            return;
        }
        
        for(int i = index; i<len;i++){
            swap(nums[i],nums[index]);
            backtrack(nums,index+1,res,len);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        int len = nums.size();
        backtrack(nums,0,ans,len);
        return ans;
    }

    //49. 字母异位词分组
    vector<vector<string>> groupAnagrams(vector<string>& strs) {        
        map < vector<int> ,vector<string>> res;
        for(int i=0;i<strs.size();i++){
            vector<int>  tmp(26,0);
            for(int j=0;j<strs[i].size();j++){
                char c = strs[i][j];
                int index = c-'a';
                tmp[index] ++;
            }
            vector<string> lookup = res[tmp];
            lookup.push_back(strs[i]);
            res[tmp] = lookup;
        }
        // for(auto val : fingers) cout<<val<<endl;
        vector<vector<string>> ans;
        map< vector<int> ,vector<string>>::iterator itr;
        itr = res.begin();
        while(itr!=res.end()){
            ans.push_back(itr->second);
            itr++;

        }
        return ans;
    }

    //55. 跳跃游戏
    bool canJump(vector<int>& nums) {
        if(nums.size()==1) return true;
        int limit=nums[0];
        int jump = 0;
        int len = nums.size();
        for(int i = 0; i <= limit; i++){
            jump = i + nums[i];
            if(jump > limit) limit = jump;
            if(limit >= len-1) return true;
        }
        return false;
    }
    //56. 合并区间
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),[](vector<int> a,vector<int> b){return a[0]<b[0];});
        // printD2Vec(intervals);
        vector<vector<int>> ans;
        if(intervals.size()==1){
            return intervals;
        }
        int begin,end;
        int delta = 0;
        for(int i=0;i<intervals.size();i++){
            if(i==intervals.size()-1){
                if(delta == 0){
                    ans.push_back({intervals[i][0],intervals[i][1]});
                }else{
                    ans.push_back({begin,end});
                }
                break;
            }
            if(delta == 0){
                begin = intervals[i][0];
                end   = intervals[i][1];
                delta = 1;
            }
            if(end<intervals[i+1][0]){
                ans.push_back({begin,end});
                delta = 0;
            } else if(end>=intervals[i+1][0] && end<=intervals[i+1][1]){
                end = intervals[i+1][1];   
            }
        }

        return ans;
    }

    //62. 不同路径
    int uniquePaths(int m, int n) {
        int r = min(m,n)-1;
        long up = 1,down = 1;
        for(int i = 0;i<r;i++){
            up = up * (m+n-2-i);
            down = down * (1+i);
        }
        return up / down;

    }
    //64.最小路径和-回溯法超时间限制
    // int m,n;
    // int backtrack(int acc ,vector<vector<int>>& grid,int i,int j){
    //     if((i+j) == (m+n-2)){
    //         if(i!=(m-1) || j!=(n-1)){
    //             printf("error run ");
    //             return 0;
    //         }
    //         return acc + grid[i][j];
    //     }else{
    //         if(i==(m-1))
    //             return backtrack(acc+grid[i][j],grid,i,j+1);
    //         else if(j==(n-1))
    //             return backtrack(acc+grid[i][j],grid,i+1,j);
    //         else
    //             return min(backtrack(acc+grid[i][j],grid,i+1,j),backtrack(acc+grid[i][j],grid,i,j+1));
    //     }
        
    // }
    // int minPathSum(vector<vector<int>>& grid) {
    //     m = grid.size();
    //     n = grid[0].size();
    //     int ans = backtrack(0,grid,0,0);
    //     return ans;
    // }
    //64.最小路径和-动态规划
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size()-1,n = grid[0].size()-1;
        // vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        // dp[m][n] = grid[m][n];
        for(int i = m-1;i>=0;i--){
            grid[i][n] = grid[i+1][n]+grid[i][n];
        }
        for(int i = n-1;i>=0;i--){
            grid[m][i] = grid[m][i+1]+grid[m][i];
        }
        for(int i = m-1;i>=0;i--){
            for(int j = n-1;j>=0;j--){
                grid[i][j] = min(grid[i+1][j],grid[i][j+1])+grid[i][j];
            }
        }
        // printD2Vec(dp);
        return grid[0][0];
    }

    //78.子集
    vector<vector<int>> subsets(vector<int>& nums) {
        if(nums.size()==1) return {{},nums};
        vector<vector<int>> ans;
        ans.push_back({});
        for(int k = 0; k<nums.size();k++){
            int n = ans.size();
            for(int i = 0;i<n;i++){
                vector<int> tmp = ans[i];
                tmp.push_back(nums[k]);
                ans.push_back(tmp);
            }
        }
        return ans;
    }

    //406. 根据身高重建队列
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(),people.end(),[](vector<int> a,vector<int> b){return (a[0]<b[0] || (a[0]==b[0] && a[1]>b[1]));});
        int n = people.size();
        vector<vector<int>> ans(n);
        for(auto person : people){
            int space = person[1]+1;
            for(int i = 0;i<n;i++){
                if(ans[i].empty())
                    space--;
                if(!space){
                    ans[i] = person;
                    break;
                }
            }
        }
        return ans;
    }

    //114. 二叉树展开为链表
    void preFlatten(TreeNode* node, vector<TreeNode*>& vec){
        if(node==nullptr) return ;
        vec.emplace_back(node);
        preFlatten(node->left,vec);
        preFlatten(node->right,vec);
        return;
    }
    void flatten(TreeNode* root) {
        if(root==nullptr) return;
        vector<TreeNode*> nodes;
        preFlatten(root,nodes);
        for(int i=0;i<nodes.size()-1;i++){
            nodes[i]->left = nullptr;
            nodes[i]->right = nodes[i+1];
        }
    }
    //238. 除自身以外数组的乘积-不能用除法而且空间复杂度为O(1)
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> L(n,0);
        vector<int> R(n,0);
        int Lp = 1,Rp = 1;
        for(int i = 0;i<n;i++){
            Lp = Lp * nums[i];          
            L[i] = Lp;
        }
        for(int i = n-1;i>=0;i--){
            Rp = Rp * nums[i];
            R[i] = Rp;
        }
        for(auto v:L) cout<<v<<" ";
        cout<<endl;
        for(auto v:R) cout<<v<<" ";
        cout<<endl;
        vector<int> ans;
        ans.push_back(R[1]);
        for(int i = 1; i<n-1;i++){
            ans.push_back(L[i-1]*R[i+1]);
        }
        ans.push_back(L[n-2]);
        return ans;
    }

    //105. Construct Binary Tree from Preorder and Inorder Traversal
    TreeNode* cirBuildTree(vector<int>& preorder, vector<int>& inorder,int preLeft,int preRight,int inLeft,int inRight){
        if(preLeft>preRight) return nullptr;

        TreeNode* root = new TreeNode(preorder[preLeft]);

        int findindex = index[preorder[preLeft]];
        int leftLen = findindex-inLeft;
        root->left = cirBuildTree(preorder,inorder,preLeft+1,preLeft+leftLen,inLeft,findindex-1);
        int RightLen = inRight-findindex;
        root->right = cirBuildTree(preorder,inorder,preLeft+leftLen+1,preRight,findindex+1,findindex+RightLen);
        return root;
    }

    unordered_map<int,int> index;
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for(int i=0;i<preorder.size();i++){
            index[inorder[i]] = i;
        }
        TreeNode* root = cirBuildTree(preorder,inorder,0,preorder.size()-1,0,inorder.size()-1);
        return root;
    }

    //96. Unique Binary Search Trees
    int numTrees(int n) {
        vector<int> factorial(n+1,1);
        factorial[0] = 1;
        factorial[1] = 1;
        factorial[2] = 2;
        factorial[3] = 5;
        if(n<=3) return factorial[n];
        for(int i=4;i<=n;i++){
            int tmpAns{0};
            for(int j=1;j<=i;j++){
                tmpAns = tmpAns+factorial[j-1]*factorial[i-j];
            }
            factorial[i] = tmpAns;
        }
        return factorial[n];
    }

    //538. Convert BST to Greater Tree
    void backOrder(TreeNode* node,int& acc){
        if(node==nullptr) return ;
        backOrder(node->right,acc);
        acc += node->val;
        node->val = acc;
        backOrder(node->left,acc);
        return ;
    }
    TreeNode* convertBST(TreeNode* root) {
        int acc{0};
        backOrder(root,acc);
        return root;
    }

    //236. Lowest Common Ancestor of a Binary Tree
    bool backOrder(TreeNode* node){
        if(node == nullptr) return false;
        bool left =  backOrder(node->left);
        bool right = backOrder(node->right);
        if(node->val==p->val || node->val == q->val){
            ancestor = node;
            return true;
        }
        if(left & right){
            ancestor = node;
            return true;
        }
        if(left | right)
            return true;
        return false;
    }
    TreeNode* p;
    TreeNode* q;
    TreeNode* ancestor;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        this->p = p;
        this->q = q;
        backOrder(root);
        return ancestor;
    }

    //287. Find the Duplicate Number
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        vector<int> index(n-1,0);
        for(auto num:nums){
            if(++index[num]>1) return num;
        }
        return 0;

    }

    //148. Sort List
    // ListNode* findMidle(ListNode* head){
    //     ListNode* fast = head;
    //     ListNode* slow = head;
    //     while (fast!=nullptr)
    //     {
    //         fast = fast->next;
    //         slow = slow->next;
    //         if(fast!=nullptr)
    //             fast = fast->next;
    //     }
    //     return slow;
    // }
    // ListNode* mergeList(ListNode* L1,ListNode* L2){
    //     ListNode* temp1 = L1;
    //     ListNode* temp2 = L2;
    //     while(temp1!=nullptr && temp2!=nullptr){
    //         if(temp1>temp2){
    //             // int tmp = temp1->val;
    //             // temp1->val = temp2->val;
    //             // temp2->val = temp1->val;
    //             swap(temp1->val,temp2->val);
                
    //         }

    //     }
    // }

    // ListNode* sortList(ListNode* head) {
    //     ListNode* mid = findMidle(head);
    //     cout<<mid->val;
    //     return head;
    // }

    //739. Daily Temperatures-暴力
    // vector<int> dailyTemperatures(vector<int>& temperatures) {
    //     vector<int> ans;
    //     int n= temperatures.size();
    //     for(int i=0;i<n;i++){
    //         int curr = temperatures[i];
    //         int j=i+1;
    //         for(;j<n;j++){
    //             if(temperatures[j]>curr){
    //                 ans.push_back(j-i);
    //                 break;
    //             }
    //         }
    //         if(j>=n) ans.push_back(0);
    //     }
    //     return ans;
    // }
    //单调栈
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> ans(n,0);
        stack<int> s;
        for(int i=0;i<n;i++){
            while(!s.empty() && (temperatures[i]>temperatures[s.top()])){
                int index = s.top();
                ans[index] = i-index;
                s.pop();
            }
            s.push(i);
        }
        return ans;

    }

    //647. Palindromic Substrings
    // int countSubstrings(string s) {

    // }

    //215. Kth Largest Element in an Array
    int findKthLargest(vector<int>& nums, int k) {
        //1//stl的sort函数
        // sort(nums.begin(),nums.end(),greater<>());
        // return nums[k-1];
        //2//构建最大堆
        return 0;
    }

    //279. Perfect Squares-动态规划
    int numSquares(int n) {
        vector<int> res(n+1,0);
        for(int i=1;i<=n;i++){
            int tmpAns=n+1;
            for(int j=1;j*j<=i;j++){
                tmpAns = min(res[i-j*j],tmpAns);
            }
            res[i] = 1+tmpAns;
        }
        return res[n];
    }

    //347. Top K Frequent Elements
    bool cmp(const pair<int,int>a,const pair<int,int>b){
        return (a.second>b.second);
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
        map<int,int> cnt;
        for(auto n :nums){
            cnt[n] ++;
        }

        vector<pair<int,int>> sortVec;
        for(auto itr=cnt.begin();itr!=cnt.end();itr++){
            sortVec.push_back(make_pair(itr->first,itr->second));
        }
        
        sort(sortVec.begin(),sortVec.end(),
        [](const pair<int,int>a,const pair<int,int>b){return a.second>b.second;});
        
        vector<int> ans;
        auto itr = sortVec.begin();
        while(k--){
            ans.push_back(itr->first);
            itr++;
        }
        return ans;
    }

    //102. 二叉树的层序遍历
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(!root){
            return ans;
        }
        //to do list：在leetcode界面上写的
        queue<TreeNode*> nodeQue;
        nodeQue.emplace(root);
        while(!nodeQue.empty()){
            TreeNode* tmp = nodeQue.front();
            cout<<tmp->val;

        }
        return ans;
    }

    //309. 最佳买卖股票时机含冷冻期
    int maxProfit(vector<int>& prices) {
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

    //337. 打家劫舍 III 

    vector<int> backDfs(TreeNode* node){
        if(!node){
            return {0,0};
        }
            
        vector<int> l = backDfs(node->left);
        vector<int> r = backDfs(node->right);
        
        int selectSelf = node->val + l[1] + r[1];//选择  本身
        int notSelectSelf = max(l[0],l[1])+max(r[0],r[1]);
        return {selectSelf,notSelectSelf};
    }

    int rob(TreeNode* root) {
        //层数遍历 分别计算 奇数层 和 偶数层的总和-
        //这个思路是错的 因为不一定非得是间隔每层选取。 
        // int ans1{0},ans2{0};
        // if(!root)
        //     return 0;
        // queue<TreeNode*> nodeQue;
        // nodeQue.emplace(root);

        // bool currLevel{true};
        // while(!nodeQue.empty()){
        //     int cnt = nodeQue.size();
        //     // cout<<" cnt: "<<cnt<<" ";
        //     while(cnt--){
        //         TreeNode* currNode = nodeQue.front();
        //         nodeQue.pop();
        //         if(currLevel) ans1+=currNode->val;
        //         else ans2+=currNode->val;
        //         if(currNode->left) nodeQue.emplace(currNode->left);
        //         if(currNode->right) nodeQue.emplace(currNode->right);
        //     }
        //     // cout<<"ans1: "<<ans1<<" ans2: "<<ans2<<endl;
        //     currLevel = !currLevel;
        // }
        // return max(ans1,ans2);

        //动态规划 - DFS遍历
        // 1.本身选择，node->val + dp(node->left->notSelect)+dp(node->right->notselect)
        // 2.本身不选择，max(dp(node->left->select),dp(node->left->notselect))  +
        //              max(dp(node->right->select),dp(node->right->notselect))
        vector<int> ans =  backDfs(root);
        return max(ans[0],ans[1]);
    }

    //75. 颜色分类
    void sortColors(vector<int>& nums) {
        int n=nums.size();
        int l{0},r{n-1};
        int c{0};
        while(c<=r && l<r){
            if(nums[c]==0){
                swap(nums[c],nums[l]);
                l++;
            }else if(nums[c]==2){

                swap(nums[c],nums[r]);
                r--;
            }else{
                c++;
            }
        }
        return;
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
    
    //
    // vector<int> nums{1,2,3,4};
    // int ans;
    // ans = s.findUnsortedSubarray(nums);
    // cout<<ans<<endl;

    //39. 组合总和
    // vector<int> vec{1};
    // vector<vector<int>> ans = s.combinationSum(vec,2);
    // // cout<<ans.size();
    // for(int i=0;i<ans.size();i++){
    //     for(int j=0;j<ans[i].size();j++){
    //         cout<<ans[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // }

    //46. 全排列
    // vector<int> vec{1,2,3};
    // vector<vector<int>> ans = s.permute(vec);
    // printD2Vec<int>(ans);

    //49. 字母异位词分组
    // vector<string> vec{"eat", "tea", "tan", "ate", "nat", "bat"};
    // vector<vector<string>> ans = s.groupAnagrams(vec);
    // printD2Vec(ans);

    //55. 跳跃游戏
    // vector<int> vec{1,2,3};
    // cout<<s.canJump(vec);

    //56. 合并区间
    // vector<vector<int>> vec{{8,10},{1,3},{2,6},{15,18}};
    // vector<vector<int>> ans =  s.merge(vec);
    // printD2Vec(ans);

    //62. 不同路径
    // int ans = s.uniquePaths(3,7);
    // cout<<ans;

    //64.最小路径和
    // vector<vector<int>> vec{{1,3,1},{1,5,1},{4,2,1}};
    // int ans = s.minPathSum(vec);
    // cout << ans;
    
    //78.子集
    // vector<int> vec{1,2,3};
    // vector<vector<int>> ans = s.subsets(vec);
    // printD2Vec(ans);

    //406. 根据身高重建队列
    // vector<vector<int>> ans;
    
    // //114. 二叉树展开为链表
    // vector<int> vec{1,2,3,4,5,-1,6};
    // TreeNode* root = createTree(vec);
    
    //238. 除自身以外数组的乘积-不能用除法而且空间复杂度为O(1)
    // vector<int> vec{4,3,2,1,2};
    // vector<int> ans = s.productExceptSelf(vec);
    // for(auto v : ans) cout<<v<<" ";

    //208. Implement Trie (Prefix Tree)
    // Trie* trie = new Trie();
    // trie->insert("apple");
    // // trie->printTrie();
    // cout<<trie->search("apple")<<endl;   // return True
    // cout<<trie->search("app")<<endl;     // return False
    // cout<<trie->startsWith("app")<<endl; // return True
    // trie->insert("app");
    // cout<<trie->search("app")<<endl;     // return True

    //105. Construct Binary Tree from Preorder and Inorder Traversal
    // vector<int> preOder{3,9,20,15,7};
    // vector<int> inOrder{9,3,15,20,7};
    // TreeNode* root = s.buildTree(preOder,inOrder);
    // printTree(root);

    //96. Unique Binary Search Trees
    // int n = 1;
    // cout<<s.numTrees(n);

    //538. Convert BST to Greater Tree
    // vector<int> vec{4,1,6,0,2,5,7,-1,-1,-1,3,-1,-1,-1,8};
    // TreeNode* root =  createTree(vec);
    // root = s.convertBST(root);
    // printTree(root);

    //236. Lowest Common Ancestor of a Binary Tree
    // vector<int> vals{3,5,1,6,2,0,8,-1,-1,7,4,-1,-1,-1,-1};
    // TreeNode* root =  createTree(vals);
    // TreeNode* p = root->left;
    // TreeNode* q = root->right;
    // TreeNode* ans = s.lowestCommonAncestor(root,p,q);
    // cout<<ans->val;

    //287. Find the Duplicate Number
    // vector<int> nums{1,3,4,2,2};
    // cout<<s.findDuplicate(nums);

    //148. Sort List
    // vector<int> vals{-1,5,3,4,0};
    // ListNode* head =  creatList(vals);
    // s.sortList(head);

    //21.Merge Two Sorted List
    // vector<int> l1Vals{2};
    // vector<int> l2Vals{1};
    // ListNode* l1 = creatList(l1Vals);
    // ListNode* l2 = creatList(l2Vals);
    // ListNode* head = s.mergeTwoLists(l1,l2);
    // printList(head);

    //739. Daily Temperatures-暴力
    // vector<int> l1Vals{73,74,75,71,69,72,76,73};
    // vector<int> ans;
    // ans = s.dailyTemperatures(l1Vals);
    // for(auto v : ans) cout<<v<<" ";

    //647. Palindromic Substrings
    // vector<int> l1Vals{3,2,1,5,6,4};
    // int ans = s.findKthLargest(l1Vals,2);
    // cout<<ans;

    //279. Perfect Squares
    // int n=12;
    // cout<<s.numSquares(12);

    //347. Top K Frequent Elements
    // vector<int> nums{-1,-1};
    // vector<int> ans = s.topKFrequent(nums,1);
    // for(auto v : ans) cout<<v<<" ";

    //102. 二叉树的层序遍历

    //337. 打家劫舍 III
    // vector<int> vec{4,1,-1,2,-1,3};
    // TreeNode* root = createTree(vec);
    // printTree(root);
    // cout<<s.rob(root);

    //75.颜色分类
    vector<int> vec{2,0,2,1,1,0};
    s.sortColors(vec);
    for_each(vec.begin(),vec.end(),[](int v){cout<<v<<"\t";});


}