const { expect } = require("@jest/globals")
const dateutils = require('./dateutils')

test('fastCountDays2 for January 2022 is correct', () => {
  from = new Date(2022, 0, 1)
  to = new Date(2022, 0, 31)
  expect(dateutils.fastCountDays2(from, to)).toEqual({
    days: 31,
    monday: 5,
    tuesday: 4,
    wednesday: 4,
    thursday: 4,
    friday: 4,
    saturday: 5,
    sunday: 5
  })
})
test('fastCountDays2 for February 2022 is correct', () => {
  from = new Date(2022, 1, 1)
  to = new Date(2022, 1, 28)
  expect(dateutils.fastCountDays2(from, to)).toEqual({
    days: 28,
    monday: 4,
    tuesday: 4,
    wednesday: 4,
    thursday: 4,
    friday: 4,
    saturday: 4,
    sunday: 4
  })
})
test('fastCountDays2 for March 2022 is correct', () => {
  from = new Date(2022, 2, 1)
  to = new Date(2022, 2, 31)
  expect(dateutils.fastCountDays2(from, to)).toEqual({
    days: 31,
    monday: 4,
    tuesday: 5,
    wednesday: 5,
    thursday: 5,
    friday: 4,
    saturday: 4,
    sunday: 4
  })
})