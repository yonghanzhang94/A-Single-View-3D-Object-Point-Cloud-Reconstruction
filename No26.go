package main

import "fmt"

func removeDuplicates(nums []int) int {
	if len(nums) > 0{
		temp_var := nums[0]
		temp_index := 0
		for i:=1; i<len(nums); i++ {
			if nums[i] != temp_var {
				temp_var = nums[i]
				temp_index++
				nums[temp_index] = temp_var
			}
		}
		nums = nums[0: temp_index+1]
	}
	return len(nums)
}

func main() {
	//nums := []int{1,1,2}
	//nums := []int{0,0,1,1,1,2,2,3,3,4}
	nums := []int{}  // 有空输入的情况

	ret := removeDuplicates(nums)
	fmt.Println(ret)
	fmt.Println(nums)
}
